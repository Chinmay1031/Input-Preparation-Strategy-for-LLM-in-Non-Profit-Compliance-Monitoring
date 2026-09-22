
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

FLAGGED_LABELS = {"FLAG", "ESCALATE"}
NA_LABEL = "N/A"

DOC_ID_MAP = {
    "WaterOrg_AuditedFinancials_2024":
        "2024_Water.org_audited_financials",
    "CARE_USA_AuditedFinancials_2024":
        "2024-CARE-USA-Financial-Statements_Final",
    "RiverNetwork_AuditedFinancials_2023":
        "8392233-FinancialStatement-1708722656073",
    "BRAC_Liberia_AuditedFinancials_2023":
        "BRAC-Liberia-Audited-Financial-Statements",
    "BRAC_Uganda_AuditedFinancials_2023":
        "BRAC-Uganda-Audited-Financial-Statements",
    "SaveTheChildrenFederation_AuditedFinancials_2024":
        "financial-statements-2024",
    "RockingTheBoat_AuditedFinancials_2023":
        "FY23_Audit",
    "AsianLawCaucus_AuditedFinancials_2025":
        "FY24-25-ALC-Audit-Financial",
    "JusticeInAging_AuditedFinancials_2023":
        "Justice-in-Aging-FY23-Audited-Financial-Statements",
    "PATH_AnnualReport_2024":
        "PATH-annual-report-2024",
    "PublicCitizenFoundation_AuditedFinancials_2024":
        "Public-Citizen-Foundation-Inc.-FS-3",
    "SOMOSMayfair_AuditedFinancials_2024":
        "Somos+2024+Audited+Financial+Statements+-+Final",
}

COLUMN_ALIASES = {
    "doc_id":                  ["doc_id", "document_id"],
    "revenue_concentration":   ["revenue_concentration",
                                 "revenue_concentration_label"],
    "expense_spike":           ["expense_spike",
                                 "expense_spike_label"],
    "passthrough_risk":        ["passthrough_risk",
                                 "pass_through_risk_label",
                                 "passthrough_risk_label"],
    "unallowable_expenditure": ["unallowable_expenditure",
                                 "unallowable_expenditure_label"],
    "audit_opinion":           ["audit_opinion",
                                 "audit_opinion_label"],
    "going_concern":           ["going_concern",
                                 "going_concern_label"],
}


def _find_column(df: pd.DataFrame, target: str) -> str:
    for alias in COLUMN_ALIASES.get(target, [target]):
        if alias in df.columns:
            return alias
    raise ValueError(
        f"Could not find column for '{target}'. "
        f"Available columns: {list(df.columns)}"
    )


def load_gold_standard(
    path: str = "data/gold_standard/gold_standard.csv"
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Gold standard not found at {path}."
        )

    df = pd.read_csv(path, quotechar='"', skipinitialspace=True)

    clean = pd.DataFrame()

    doc_col = _find_column(df, "doc_id")
    clean["doc_id"] = df[doc_col].astype(str).str.strip()

    for dim in DIMENSIONS:
        col = _find_column(df, dim)
        clean[dim] = df[col].astype(str).str.strip().str.upper()

    clean["doc_id"] = clean["doc_id"].apply(
        lambda x: DOC_ID_MAP.get(x, x)
    )

    return clean


def get_binary_labels(df: pd.DataFrame, doc_id: str) -> dict:
    row = df[df["doc_id"] == doc_id]
    if row.empty:
        raise ValueError(
            f"Document '{doc_id}' not found in gold standard. "
            f"Available: {df['doc_id'].tolist()}"
        )

    labels = {}
    for dim in DIMENSIONS:
        val = str(row.iloc[0][dim]).strip().upper()
        if val in (NA_LABEL, "NAN", "NA", ""):
            labels[dim] = None
        else:
            labels[dim] = 1 if val in FLAGGED_LABELS else 0

    return labels


def get_all_doc_ids(
    path: str = "data/gold_standard/gold_standard.csv"
) -> list:
    df = load_gold_standard(path)
    return df["doc_id"].tolist()