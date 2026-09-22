import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from analysis.config import (
    DIMENSIONS,
    EXPERIMENT_DIMENSION_MAP,
    POSITIVE_LABELS,
    STRATEGIES,
    STRATEGY_LABELS,
    QUALITY_DIR,
)

from analysis.validation import validate_dataset


def to_binary(label):
    """
    Convert a compliance label into the binary evaluation format.

    Positive:
        FLAG
        ESCALATE

    Negative:
        CLEAR
        N/A

    Gold-standard N/A values are excluded before this function
    is called. Predicted N/A values are treated as non-flagged,
    matching the original experiment scoring implementation.
    """

    if label in POSITIVE_LABELS:
        return 1

    if label in {"CLEAR", "N/A"}:
        return 0

    raise ValueError(
        f"Unexpected label encountered: {label}"
    )


def build_quality_dataset(records, gold_df):
    """
    Build the run-0 evaluation dataset.

    Each row represents one:
        document × strategy × compliance dimension

    Gold-standard N/A judgements are excluded.

    Predicted N/A judgements are also excluded because N/A
    does not represent either a positive or negative compliance
    classification and therefore cannot be converted to the
    binary FLAG/ESCALATE vs CLEAR evaluation.

    The gold-standard dimension names and experiment-result
    dimension names are mapped through
    EXPERIMENT_DIMENSION_MAP.
    """

    gold_lookup = gold_df.set_index("document_id")

    rows = []

    for record in records:

        if record.get("run") != 0:
            continue

        experiment_doc_id = record["doc_id"]
        gold_doc_id = record["gold_document_id"]
        strategy = record["strategy"]

        if gold_doc_id not in gold_lookup.index:
            raise ValueError(
                f"Gold document not found: {gold_doc_id}"
            )

        gold_row = gold_lookup.loc[gold_doc_id]

        for dimension in DIMENSIONS:

            gold_column = f"{dimension}_label"

            if gold_column not in gold_df.columns:
                raise ValueError(
                    f"Gold-standard column "
                    f"'{gold_column}' is missing."
                )

            experiment_dimension = (
                EXPERIMENT_DIMENSION_MAP.get(dimension)
            )

            if experiment_dimension is None:
                raise ValueError(
                    f"No experiment dimension mapping "
                    f"exists for '{dimension}'."
                )

            if experiment_dimension not in record:
                raise ValueError(
                    f"Experiment field "
                    f"'{experiment_dimension}' is missing "
                    f"from experiment record."
                )

            predicted_label = record[
                experiment_dimension
            ]

            gold_label = gold_row[
                gold_column
            ]

            if pd.isna(gold_label):
                continue

            gold_label = str(
                gold_label
            ).strip()

            if gold_label == "N/A":
                continue

            if pd.isna(predicted_label):
                continue

            predicted_label = str(
                predicted_label
            ).strip()

            rows.append({
                "experiment_doc_id": experiment_doc_id,
                "gold_document_id": gold_doc_id,
                "strategy": strategy,
                "dimension": dimension,
                "experiment_dimension": experiment_dimension,
                "gold_label": gold_label,
                "predicted_label": predicted_label,
                "gold_binary": to_binary(
                    gold_label
                ),
                "predicted_binary": to_binary(
                    predicted_label
                ),
            })

    evaluation_df = pd.DataFrame(rows)

    if evaluation_df.empty:
        raise ValueError(
            "The quality evaluation dataset is empty."
        )

    return evaluation_df


def calculate_overall_quality(evaluation_df):
    """
    Calculate overall binary classification metrics for
    each strategy.
    """

    results = []

    for strategy in STRATEGIES:

        subset = evaluation_df[
            evaluation_df["strategy"] == strategy
        ].copy()

        if subset.empty:
            raise ValueError(
                f"No evaluable observations for {strategy}."
            )

        y_true = subset["gold_binary"]
        y_pred = subset["predicted_binary"]

        precision = precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        correct = int(
            (y_true == y_pred).sum()
        )

        results.append({
            "strategy": strategy,
            "strategy_label": STRATEGY_LABELS[strategy],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "judgements": len(subset),
            "correct": correct,
        })

    return pd.DataFrame(results)


def calculate_dimension_quality(evaluation_df):
    """
    Calculate quality metrics separately for each
    compliance dimension and strategy.
    """

    results = []

    for strategy in STRATEGIES:

        for dimension in DIMENSIONS:

            subset = evaluation_df[
                (evaluation_df["strategy"] == strategy)
                & (
                    evaluation_df["dimension"]
                    == dimension
                )
            ].copy()

            if subset.empty:
                continue

            y_true = subset["gold_binary"]
            y_pred = subset["predicted_binary"]

            results.append({
                "strategy": strategy,
                "strategy_label": STRATEGY_LABELS[strategy],
                "dimension": dimension,
                "dimension_label": (
                    dimension
                    .replace("_", " ")
                    .title()
                ),
                "precision": precision_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),
                "accuracy": accuracy_score(
                    y_true,
                    y_pred,
                ),
                "judgements": len(subset),
                "correct": int(
                    (y_true == y_pred).sum()
                ),
            })

    return pd.DataFrame(results)


def run_quality_analysis():

    records, gold_df = validate_dataset()

    print("\n" + "=" * 60)
    print("QUALITY ANALYSIS")
    print("=" * 60)

    evaluation_df = build_quality_dataset(
        records,
        gold_df,
    )

    print("\nEVALUATION DATA")
    print("-" * 60)

    print(
        f"Evaluable run-0 judgements: "
        f"{len(evaluation_df)}"
    )

    print(
        f"Unique documents: "
        f"{evaluation_df['gold_document_id'].nunique()}"
    )

    print(
        f"Strategies: "
        f"{evaluation_df['strategy'].nunique()}"
    )

    print(
        f"Dimensions: "
        f"{evaluation_df['dimension'].nunique()}"
    )

    overall_df = calculate_overall_quality(
        evaluation_df
    )

    dimension_df = calculate_dimension_quality(
        evaluation_df
    )

    QUALITY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_file = (
        QUALITY_DIR
        / "evaluation_dataset.csv"
    )

    overall_file = (
        QUALITY_DIR
        / "overall_quality.csv"
    )

    dimension_file = (
        QUALITY_DIR
        / "dimension_quality.csv"
    )

    evaluation_df.to_csv(
        evaluation_file,
        index=False,
    )

    overall_df.to_csv(
        overall_file,
        index=False,
    )

    dimension_df.to_csv(
        dimension_file,
        index=False,
    )

    print("\n" + "=" * 60)
    print("OVERALL QUALITY")
    print("=" * 60)

    print(
        overall_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\n" + "=" * 60)
    print("DIMENSION-LEVEL QUALITY")
    print("=" * 60)

    print(
        dimension_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\n" + "=" * 60)
    print("FILES SAVED")
    print("=" * 60)

    print(evaluation_file)
    print(overall_file)
    print(dimension_file)

    print("\n" + "=" * 60)
    print("QUALITY ANALYSIS COMPLETE")
    print("=" * 60)

    return (
        evaluation_df,
        overall_df,
        dimension_df,
    )


if __name__ == "__main__":
    run_quality_analysis()