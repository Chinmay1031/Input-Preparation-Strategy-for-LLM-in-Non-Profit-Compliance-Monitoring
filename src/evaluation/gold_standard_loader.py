"""
gold_standard_loader.py
-----------------------
Loads the manually annotated gold standard CSV and converts
labels to binary format for metric computation.

Gold standard is the answer key for the entire experiment.
Every LLM output is measured against these human judgements.
"""

import pandas as pd
from pathlib import Path

DIMENSIONS = [
    "revenue_concentration",
    "expense_spike",
    "passthrough_risk",
    "unallowable_expenditure",
    "audit_opinion",
    "going_concern",
]

# Labels that count as flagged (binary 1)
FLAGGED_LABELS = {"FLAG", "ESCALATE"}

# N/A dimensions are excluded from F1 computation
NA_LABEL = "N/A"

# The annotations use concise source-document identifiers, while the pipeline
# uses filenames as document IDs.  Normalise the two existing annotations at
# import so quality metrics can join them reliably.
DOCUMENT_ID_ALIASES = {
    "EduCon_AUP_2024": "Audit_2024_EduCon.HPF.SPAC.250513.Final",
    "SPAC_NPC_FinancialStatements_2024": (
        "FinancialStatements_2024_signed.HPF.SPAC.250604.Draft"
    ),
}


def load_gold_standard(
    path: str = "data/gold_standard/gold_standard.csv"
) -> pd.DataFrame:
    """
    Load gold standard CSV and return as DataFrame.
    Validates that all required columns are present.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Gold standard not found at {path}. "
            f"Create it before running evaluation."
        )

    # The annotated source file is a semicolon-delimited CSV and may contain
    # a UTF-8 byte-order mark when exported from spreadsheet software.
    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    df["doc_id"] = df["doc_id"].str.strip().replace(DOCUMENT_ID_ALIASES)

    # Validate columns
    missing = [d for d in DIMENSIONS if d not in df.columns]
    if missing:
        raise ValueError(
            f"Gold standard missing columns: {missing}"
        )

    return df


def get_binary_labels(
    df: pd.DataFrame,
    doc_id: str
) -> dict:
    """
    Get binary labels for one document from the gold standard.
    Returns dict of dimension -> 0 or 1.
    N/A dimensions return None and are excluded from scoring.
    """
    row = df[df["doc_id"] == doc_id]
    if row.empty:
        raise ValueError(
            f"Document '{doc_id}' not found in gold standard."
        )

    labels = {}
    for dim in DIMENSIONS:
        val = str(row.iloc[0][dim]).strip().upper()
        if val == NA_LABEL:
            labels[dim] = None  # excluded from scoring
        else:
            labels[dim] = 1 if val in FLAGGED_LABELS else 0

    return labels


def get_all_doc_ids(
    path: str = "data/gold_standard/gold_standard.csv"
) -> list:
    """Return list of all document IDs in the gold standard."""
    df = load_gold_standard(path)
    return df["doc_id"].tolist()
