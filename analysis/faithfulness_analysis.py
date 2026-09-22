import sys
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.config import (
    STRATEGIES,
    STRATEGY_LABELS,
    FAITHFULNESS_DIR,
)

from analysis.validation import validate_dataset

from src.ingestion import parse_document
from src.strategies import get_all_strategies

from src.output_processing.result_schema import (
    ExperimentResult,
    ComplianceFlag,
)

from src.output_processing import compute_faithfulness

FAITHFULNESS_RUN = 0

RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "all_results.json"
)

PDF_DIR = (
    PROJECT_ROOT
    / "data"
    / "pdfs"
)

def load_raw_results():
    """Load the raw experiment results; the file itself is never modified."""

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_PATH}"
        )

    with open(
        RESULTS_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "Expected all_results.json to contain a list."
        )

    return data

def build_document_cache():
    """Parse each PDF once and reconstruct the strategy-specific prepared contexts."""

    if not PDF_DIR.exists():
        raise FileNotFoundError(
            f"PDF directory not found: {PDF_DIR}"
        )

    cache = {}

    pdf_paths = sorted(
        PDF_DIR.glob("*.pdf")
    )

    if not pdf_paths:
        raise ValueError(
            f"No PDFs found in {PDF_DIR}"
        )

    print("\nDOCUMENT CACHE")
    print("-" * 60)

    for pdf_path in pdf_paths:

        doc_id = pdf_path.stem

        print(
            f"Parsing {doc_id}..."
        )

        doc = parse_document(
            str(pdf_path),
            doc_id,
        )

        strategies = get_all_strategies(
            doc
        )

        cache[doc_id] = {
            "full_text": doc.full_text,
            "strategies": strategies,
        }

    print(
        f"\nParsed documents: {len(cache)}"
    )

    return cache

def build_experiment_result(record):
    """Convert one raw JSON result into the ExperimentResult object required by compute_faithfulness()."""

    flags = []

    for flag in record.get(
        "flags",
        [],
    ):

        flags.append(
            ComplianceFlag(
                dimension=flag.get(
                    "dimension",
                    "",
                ),
                severity=flag.get(
                    "severity",
                    "FLAG",
                ),
                evidence=flag.get(
                    "evidence",
                    "",
                ),
                is_faithful=flag.get(
                    "is_faithful"
                ),
                match_score=flag.get(
                    "match_score"
                ),
            )
        )

    return ExperimentResult(
        doc_id=record["doc_id"],
        strategy=record["strategy"],
        run=record["run"],

        revenue_concentration=record.get(
            "revenue_concentration",
            "CLEAR",
        ),

        expense_spike=record.get(
            "expense_spike",
            "CLEAR",
        ),

        passthrough_risk=record.get(
            "passthrough_risk",
            "CLEAR",
        ),

        unallowable_expenditure=record.get(
            "unallowable_expenditure",
            "CLEAR",
        ),

        audit_opinion=record.get(
            "audit_opinion",
            "CLEAR",
        ),

        going_concern=record.get(
            "going_concern",
            "CLEAR",
        ),

        overall_verdict=record.get(
            "overall_verdict",
            "COMPLIANT",
        ),

        confidence=float(
            record.get(
                "confidence",
                0.0,
            )
        ),

        flags=flags,

        tokens_input=int(
            record.get(
                "tokens_input",
                0,
            )
        ),

        tokens_output=int(
            record.get(
                "tokens_output",
                0,
            )
        ),

        tokens_total=int(
            record.get(
                "tokens_total",
                0,
            )
        ),

        parse_error=record.get(
            "parse_error",
            False,
        ),

        error_message=record.get(
            "error_message",
            "",
        ),

        model=record.get(
            "model",
            "",
        ),
    )

def build_faithfulness_dataset(
    records
):
    """Build one row per FLAG/ESCALATE evidence item from Run 0."""

    rows = []

    for record in records:

        if record["run"] != FAITHFULNESS_RUN:
            continue

        strategy = record["strategy"]

        for flag in record.get(
            "flags",
            [],
        ):

            evidence = flag.get(
                "evidence",
                "",
            )

            if not evidence:
                continue

            rows.append({
                "doc_id": record["doc_id"],
                "strategy": strategy,
                "strategy_label": STRATEGY_LABELS[
                    strategy
                ],
                "run": record["run"],
                "dimension": flag.get(
                    "dimension",
                    "",
                ),
                "verdict": flag.get(
                    "severity",
                    "FLAG",
                ),
                "evidence": evidence,
            })

    df = pd.DataFrame(
        rows
    )

    if df.empty:
        raise ValueError(
            "No evidence records were found "
            "in the Run-0 experiment results."
        )

    return df

def evaluate_faithfulness(
    records,
    document_cache,
):
    """
    Recalculate faithfulness using compute_faithfulness().

    Source text is the original parsed PDF text; prepared text is the
    strategy-specific context generated from that same document.

    S1/S2 are checked against source text, while S3/S4/S5 use the
    prepared context when it's available.
    """

    rows = []

    evaluated_documents = 0
    skipped_documents = 0

    for record in records:

        if record["run"] != FAITHFULNESS_RUN:
            continue

        doc_id = record["doc_id"]
        strategy = record["strategy"]

        if doc_id not in document_cache:

            skipped_documents += 1

            continue

        cache = document_cache[
            doc_id
        ]

        result = build_experiment_result(
            record
        )

        prepared_text = (
            cache["strategies"]
            .get(
                strategy,
                {}
            )
            .get(
                "text",
                "",
            )
        )

        result = compute_faithfulness(
            result,
            source_text=cache["full_text"],
            prepared_text=prepared_text,
        )

        evaluated_documents += 1

        for flag in result.flags:

            match_score = (
                float(flag.match_score)
                if flag.match_score is not None
                else None
            )

            rows.append({
                "doc_id": result.doc_id,
                "strategy": result.strategy,
                "strategy_label": STRATEGY_LABELS[
                    result.strategy
                ],
                "run": result.run,
                "dimension": flag.dimension,
                "verdict": flag.severity,
                "evidence": flag.evidence,
                "match_score": match_score,
                "is_faithful": flag.is_faithful,
                "comparison_mode": (
                    "prepared_context"
                    if result.strategy
                    in {
                        "S3_fields",
                        "S4_hybrid",
                        "S5_extended",
                    }
                    and prepared_text
                    else "source_text"
                ),
            })

    result_df = pd.DataFrame(
        rows
    )

    print(
        f"Evaluated Run-0 records: "
        f"{evaluated_documents}"
    )

    if skipped_documents:
        print(
            f"Skipped records: "
            f"{skipped_documents}"
        )

    if result_df.empty:
        raise ValueError(
            "Faithfulness evaluation produced "
            "no flag-level results."
        )

    return result_df

def classify_evidence_type(
    match_score
):
    """Classify evidence using the faithfulness checker's thresholds (match scores are on a 0-1 scale)."""

    if pd.isna(match_score):
        return "No score"

    if match_score >= 0.92:
        return "Verbatim"

    if match_score >= 0.60:
        return "Synthesised"

    return "Unsupported"


def add_evidence_type(
    result_df
):

    result = result_df.copy()

    result["evidence_type"] = (
        result["match_score"]
        .apply(
            classify_evidence_type
        )
    )

    return result

def calculate_strategy_faithfulness(
    result_df
):

    rows = []

    for strategy in STRATEGIES:

        subset = result_df[
            result_df["strategy"] == strategy
        ]

        if subset.empty:
            continue

        total = len(
            subset
        )

        faithful_rate = (
            subset["is_faithful"]
            .astype(bool)
            .mean()
        )

        rows.append({
            "strategy": strategy,

            "strategy_label": STRATEGY_LABELS[
                strategy
            ],

            "evidence_cases": total,

            "mean_faithfulness": (
                faithful_rate
            ),

            "hallucination_rate": (
                1.0 - faithful_rate
            ),

            "verbatim_rate": (
                subset["evidence_type"]
                .eq("Verbatim")
                .mean()
            ),

            "synthesised_rate": (
                subset["evidence_type"]
                .eq("Synthesised")
                .mean()
            ),

            "unsupported_rate": (
                subset["evidence_type"]
                .eq("Unsupported")
                .mean()
            ),

            "mean_match_score": (
                subset["match_score"]
                .mean()
            ),
        })

    return pd.DataFrame(
        rows
    )

def calculate_document_faithfulness(
    result_df
):

    rows = []

    for (
        doc_id,
        strategy,
    ), group in result_df.groupby(
        [
            "doc_id",
            "strategy",
        ]
    ):

        faithful_rate = (
            group["is_faithful"]
            .astype(bool)
            .mean()
        )

        rows.append({
            "doc_id": doc_id,

            "strategy": strategy,

            "strategy_label": STRATEGY_LABELS[
                strategy
            ],

            "evidence_cases": len(
                group
            ),

            "mean_faithfulness": (
                faithful_rate
            ),

            "hallucination_rate": (
                1.0 - faithful_rate
            ),

            "verbatim_rate": (
                group["evidence_type"]
                .eq("Verbatim")
                .mean()
            ),

            "synthesised_rate": (
                group["evidence_type"]
                .eq("Synthesised")
                .mean()
            ),

            "unsupported_rate": (
                group["evidence_type"]
                .eq("Unsupported")
                .mean()
            ),

            "mean_match_score": (
                group["match_score"]
                .mean()
            ),
        })

    return pd.DataFrame(
        rows
    )

def calculate_evidence_type_summary(
    result_df
):

    rows = []

    for strategy in STRATEGIES:

        subset = result_df[
            result_df["strategy"] == strategy
        ]

        if subset.empty:
            continue

        total = len(
            subset
        )

        for evidence_type in [
            "Verbatim",
            "Synthesised",
            "Unsupported",
        ]:

            count = (
                subset["evidence_type"]
                .eq(evidence_type)
                .sum()
            )

            rows.append({
                "strategy": strategy,

                "strategy_label": STRATEGY_LABELS[
                    strategy
                ],

                "evidence_type": evidence_type,

                "count": count,

                "percentage": (
                    count / total * 100
                ),
            })

    return pd.DataFrame(
        rows
    )

def run_faithfulness_analysis():

    records, _ = validate_dataset()

    print("\n" + "=" * 60)
    print("FAITHFULNESS ANALYSIS")
    print("=" * 60)

    faithfulness_df = (
        build_faithfulness_dataset(
            records
        )
    )

    print("\nFAITHFULNESS DATA")
    print("-" * 60)

    print(
        f"Evidence records: "
        f"{len(faithfulness_df)}"
    )

    print(
        f"Documents: "
        f"{faithfulness_df['doc_id'].nunique()}"
    )

    print(
        f"Strategies: "
        f"{faithfulness_df['strategy'].nunique()}"
    )

    print(
        f"Runs evaluated: "
        f"{sorted(faithfulness_df['run'].unique())}"
    )

    document_cache = (
        build_document_cache()
    )

    result_df = evaluate_faithfulness(
        records,
        document_cache,
    )

    result_df = add_evidence_type(
        result_df
    )

    strategy_df = (
        calculate_strategy_faithfulness(
            result_df
        )
    )

    document_df = (
        calculate_document_faithfulness(
            result_df
        )
    )

    evidence_type_df = (
        calculate_evidence_type_summary(
            result_df
        )
    )

    FAITHFULNESS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_file = (
        FAITHFULNESS_DIR
        / "faithfulness_dataset.csv"
    )

    evaluated_file = (
        FAITHFULNESS_DIR
        / "faithfulness_evaluated.csv"
    )

    strategy_file = (
        FAITHFULNESS_DIR
        / "strategy_faithfulness.csv"
    )

    document_file = (
        FAITHFULNESS_DIR
        / "document_faithfulness.csv"
    )

    evidence_type_file = (
        FAITHFULNESS_DIR
        / "evidence_type_summary.csv"
    )

    faithfulness_df.to_csv(
        dataset_file,
        index=False,
    )

    result_df.to_csv(
        evaluated_file,
        index=False,
    )

    strategy_df.to_csv(
        strategy_file,
        index=False,
    )

    document_df.to_csv(
        document_file,
        index=False,
    )

    evidence_type_df.to_csv(
        evidence_type_file,
        index=False,
    )

    print("\n" + "=" * 60)
    print("STRATEGY-LEVEL FAITHFULNESS")
    print("=" * 60)

    print(
        strategy_df[
            [
                "strategy",
                "strategy_label",
                "evidence_cases",
                "mean_faithfulness",
                "hallucination_rate",
                "verbatim_rate",
                "synthesised_rate",
                "unsupported_rate",
                "mean_match_score",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\n" + "=" * 60)
    print("EVIDENCE TYPE DISTRIBUTION")
    print("=" * 60)

    print(
        evidence_type_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\n" + "=" * 60)
    print("FILES SAVED")
    print("=" * 60)

    print(dataset_file)
    print(evaluated_file)
    print(strategy_file)
    print(document_file)
    print(evidence_type_file)

    print("\n" + "=" * 60)
    print("FAITHFULNESS ANALYSIS COMPLETE")
    print("=" * 60)

    return (
        faithfulness_df,
        result_df,
        strategy_df,
        document_df,
        evidence_type_df,
    )

if __name__ == "__main__":
    run_faithfulness_analysis()