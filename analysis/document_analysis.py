import pandas as pd
from sklearn.metrics import f1_score

from analysis.config import (
    DIMENSIONS,
    STRATEGIES,
    STRATEGY_LABELS,
    DOCUMENT_DIR,
)

from analysis.validation import validate_dataset
from analysis.quality_analysis import (
    to_binary,
)


def build_document_quality_dataset(records, gold_df):
    """
    Build the run-0 document-level evaluation dataset.

    Each row represents one:

        document × strategy × compliance dimension

    Gold-standard N/A judgements are excluded.

    Predicted N/A values are treated as non-flagged (0),
    matching the original experiment scoring implementation.
    """

    gold_lookup = (
        gold_df
        .set_index("document_id")
    )

    rows = []

    for record in records:

        # Quality evaluation uses run 0 only
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

            gold_label = gold_row[gold_column]

            # Gold N/A is not evaluable.
            if pd.isna(gold_label):
                continue

            gold_label = str(
                gold_label
            ).strip()

            if gold_label == "N/A":
                continue

            from analysis.config import (
                EXPERIMENT_DIMENSION_MAP
            )

            experiment_dimension = (
                EXPERIMENT_DIMENSION_MAP[dimension]
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

            if pd.isna(predicted_label):
                raise ValueError(
                    f"Missing predicted label for "
                    f"{experiment_dimension} in "
                    f"{experiment_doc_id}, "
                    f"{strategy}."
                )

            predicted_label = str(
                predicted_label
            ).strip()

            rows.append({
                "experiment_doc_id": experiment_doc_id,
                "gold_document_id": gold_doc_id,
                "strategy": strategy,
                "dimension": dimension,
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
            "Document-level evaluation dataset is empty."
        )

    return evaluation_df


def calculate_document_f1(evaluation_df):
    """
    Calculate macro F1 separately for every document
    and strategy.

    Each document contributes one F1 value per strategy.
    """

    results = []

    documents = sorted(
        evaluation_df["gold_document_id"].unique()
    )

    for document_id in documents:

        for strategy in STRATEGIES:

            subset = evaluation_df[
                (evaluation_df["gold_document_id"] == document_id)
                & (evaluation_df["strategy"] == strategy)
            ].copy()

            if subset.empty:
                continue

            y_true = subset["gold_binary"]
            y_pred = subset["predicted_binary"]

            f1 = f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )

            results.append({
                "gold_document_id": document_id,
                "strategy": strategy,
                "strategy_label": STRATEGY_LABELS[strategy],
                "f1": f1,
                "judgements": len(subset),
                "correct": int(
                    (y_true == y_pred).sum()
                ),
            })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        raise ValueError(
            "No document-level F1 results were generated."
        )

    return result_df


def calculate_document_summary(document_f1_df):
    """
    Summarise the distribution of document-level F1 values
    for each strategy.
    """

    rows = []

    for strategy in STRATEGIES:

        subset = document_f1_df[
            document_f1_df["strategy"] == strategy
        ]["f1"]

        if subset.empty:
            continue

        rows.append({
            "strategy": strategy,
            "strategy_label": STRATEGY_LABELS[strategy],
            "documents": len(subset),
            "mean_f1": subset.mean(),
            "median_f1": subset.median(),
            "std_f1": subset.std(
                ddof=1
            ),
            "min_f1": subset.min(),
            "max_f1": subset.max(),
        })

    return pd.DataFrame(rows)


def run_document_analysis():

    records, gold_df = validate_dataset()

    print("\n" + "=" * 60)
    print("DOCUMENT-LEVEL QUALITY ANALYSIS")
    print("=" * 60)

    evaluation_df = (
        build_document_quality_dataset(
            records,
            gold_df,
        )
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

    document_f1_df = calculate_document_f1(
        evaluation_df
    )

    summary_df = calculate_document_summary(
        document_f1_df
    )

    DOCUMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_file = (
        DOCUMENT_DIR
        / "document_evaluation_dataset.csv"
    )

    document_f1_file = (
        DOCUMENT_DIR
        / "document_level_f1.csv"
    )

    summary_file = (
        DOCUMENT_DIR
        / "document_f1_summary.csv"
    )

    evaluation_df.to_csv(
        evaluation_file,
        index=False,
    )

    document_f1_df.to_csv(
        document_f1_file,
        index=False,
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    print("\n" + "=" * 60)
    print("DOCUMENT-LEVEL F1")
    print("=" * 60)

    print(
        document_f1_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\n" + "=" * 60)
    print("DOCUMENT-LEVEL F1 SUMMARY")
    print("=" * 60)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\n" + "=" * 60)
    print("FILES SAVED")
    print("=" * 60)

    print(evaluation_file)
    print(document_f1_file)
    print(summary_file)

    print("\n" + "=" * 60)
    print("DOCUMENT ANALYSIS COMPLETE")
    print("=" * 60)

    return (
        evaluation_df,
        document_f1_df,
        summary_df,
    )


if __name__ == "__main__":
    run_document_analysis()