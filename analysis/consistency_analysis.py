import pandas as pd
from sklearn.metrics import cohen_kappa_score

from analysis.config import (
    STRATEGIES,
    STRATEGY_LABELS,
    CONSISTENCY_DIR,
    EXPERIMENT_DIMENSION_MAP,
)

from analysis.validation import validate_dataset


RUN_PAIRS = [
    (0, 1),
    (0, 2),
    (1, 2),
]


def to_binary(label):
    """
    Convert a compliance label to the binary representation
    used for consistency analysis.

    FLAG / ESCALATE -> 1
    CLEAR / N/A     -> 0
    """

    if pd.isna(label):
        return 0

    label = str(label).strip().upper()

    if label in {"FLAG", "ESCALATE"}:
        return 1

    return 0


def build_consistency_dataset(records):
    """
    Build a long-format dataset containing one row for every:

        document × strategy × run × compliance dimension

    All three experimental runs are retained.
    """

    rows = []

    for record in records:

        strategy = record["strategy"]
        run = record["run"]

        for dimension in EXPERIMENT_DIMENSION_MAP:

            experiment_dimension = (
                EXPERIMENT_DIMENSION_MAP[dimension]
            )

            if experiment_dimension not in record:
                raise ValueError(
                    f"Missing dimension "
                    f"'{experiment_dimension}' in "
                    f"{record['doc_id']} / "
                    f"{strategy} / run {run}"
                )

            label = record[experiment_dimension]

            rows.append({
                "doc_id": record["doc_id"],
                "gold_document_id": record.get(
                    "gold_document_id"
                ),
                "strategy": strategy,
                "strategy_label": STRATEGY_LABELS[strategy],
                "run": run,
                "dimension": dimension,
                "label": label,
                "binary_label": to_binary(label),
            })

    consistency_df = pd.DataFrame(rows)

    if consistency_df.empty:
        raise ValueError(
            "Consistency dataset is empty."
        )

    return consistency_df


def calculate_pairwise_consistency(
    consistency_df,
    strategy,
    run_a,
    run_b,
):
    """
    Calculate Cohen's Kappa and raw agreement for one
    strategy and one pair of runs.

    The comparison is performed across all documents and
    all six compliance dimensions.
    """

    subset = consistency_df[
        consistency_df["strategy"] == strategy
    ].copy()

    run_a_df = subset[
        subset["run"] == run_a
    ][
        [
            "doc_id",
            "dimension",
            "binary_label",
        ]
    ].rename(
        columns={
            "binary_label": "label_a"
        }
    )

    run_b_df = subset[
        subset["run"] == run_b
    ][
        [
            "doc_id",
            "dimension",
            "binary_label",
        ]
    ].rename(
        columns={
            "binary_label": "label_b"
        }
    )

    merged = run_a_df.merge(
        run_b_df,
        on=[
            "doc_id",
            "dimension",
        ],
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            f"No paired observations found for "
            f"{strategy}: Run {run_a} vs Run {run_b}"
        )

    y_a = merged["label_a"]
    y_b = merged["label_b"]

    kappa = cohen_kappa_score(
        y_a,
        y_b,
    )

    agreement = (
        y_a == y_b
    ).mean()

    return {
        "strategy": strategy,
        "strategy_label": STRATEGY_LABELS[strategy],
        "run_a": run_a,
        "run_b": run_b,
        "comparison": f"Run {run_a} vs Run {run_b}",
        "observations": len(merged),
        "kappa": kappa,
        "agreement_rate": agreement,
    }


def calculate_strategy_consistency(
    consistency_df
):
    """
    Calculate all three pairwise comparisons and the
    resulting strategy-level mean Cohen's Kappa.
    """

    pairwise_rows = []

    for strategy in STRATEGIES:

        for run_a, run_b in RUN_PAIRS:

            result = calculate_pairwise_consistency(
                consistency_df,
                strategy,
                run_a,
                run_b,
            )

            pairwise_rows.append(result)

    pairwise_df = pd.DataFrame(
        pairwise_rows
    )

    summary_rows = []

    for strategy in STRATEGIES:

        subset = pairwise_df[
            pairwise_df["strategy"] == strategy
        ]

        if subset.empty:
            continue

        summary_rows.append({
            "strategy": strategy,
            "strategy_label": STRATEGY_LABELS[strategy],
            "comparisons": len(subset),
            "mean_kappa": subset["kappa"].mean(),
            "mean_agreement_rate": subset[
                "agreement_rate"
            ].mean(),
            "min_kappa": subset["kappa"].min(),
            "max_kappa": subset["kappa"].max(),
            "observations_per_comparison": subset[
                "observations"
            ].iloc[0],
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    return pairwise_df, summary_df


def interpret_kappa(kappa):
    """
    Descriptive interpretation using the commonly cited
    Landis and Koch bands.

    The numerical kappa value remains the primary result.
    """

    if kappa < 0:
        return "Less than chance"

    if kappa <= 0.20:
        return "Slight"

    if kappa <= 0.40:
        return "Fair"

    if kappa <= 0.60:
        return "Moderate"

    if kappa <= 0.80:
        return "Substantial"

    return "Almost perfect"


def add_kappa_interpretation(summary_df):

    result = summary_df.copy()

    result["kappa_interpretation"] = (
        result["mean_kappa"]
        .apply(interpret_kappa)
    )

    return result


def calculate_document_consistency(
    consistency_df
):
    """
    Calculate the mean pairwise agreement for each:

        document × strategy

    This provides a descriptive view of whether consistency
    varies across individual documents.
    """

    rows = []

    for strategy in STRATEGIES:

        for document_id in sorted(
            consistency_df[
                consistency_df["strategy"] == strategy
            ]["doc_id"].unique()
        ):

            subset = consistency_df[
                (consistency_df["strategy"] == strategy)
                & (consistency_df["doc_id"] == document_id)
            ]

            pairwise_agreements = []

            for run_a, run_b in RUN_PAIRS:

                run_a_values = subset[
                    subset["run"] == run_a
                ][
                    [
                        "dimension",
                        "binary_label",
                    ]
                ].rename(
                    columns={
                        "binary_label": "label_a"
                    }
                )

                run_b_values = subset[
                    subset["run"] == run_b
                ][
                    [
                        "dimension",
                        "binary_label",
                    ]
                ].rename(
                    columns={
                        "binary_label": "label_b"
                    }
                )

                merged = run_a_values.merge(
                    run_b_values,
                    on="dimension",
                    how="inner",
                )

                if not merged.empty:

                    agreement = (
                        merged["label_a"]
                        ==
                        merged["label_b"]
                    ).mean()

                    pairwise_agreements.append(
                        agreement
                    )

            if pairwise_agreements:

                rows.append({
                    "doc_id": document_id,
                    "strategy": strategy,
                    "strategy_label": STRATEGY_LABELS[
                        strategy
                    ],
                    "mean_pairwise_agreement": (
                        sum(pairwise_agreements)
                        /
                        len(pairwise_agreements)
                    ),
                })

    return pd.DataFrame(rows)


def run_consistency_analysis():

    records, _ = validate_dataset()

    print("\n" + "=" * 60)
    print("CONSISTENCY ANALYSIS")
    print("=" * 60)

    consistency_df = (
        build_consistency_dataset(
            records
        )
    )

    print("\nCONSISTENCY DATA")
    print("-" * 60)

    print(
        f"Records: {len(consistency_df)}"
    )

    print(
        f"Documents: "
        f"{consistency_df['doc_id'].nunique()}"
    )

    print(
        f"Strategies: "
        f"{consistency_df['strategy'].nunique()}"
    )

    print(
        f"Runs: "
        f"{consistency_df['run'].nunique()}"
    )

    print(
        f"Dimensions: "
        f"{consistency_df['dimension'].nunique()}"
    )

    pairwise_df, summary_df = (
        calculate_strategy_consistency(
            consistency_df
        )
    )

    summary_df = add_kappa_interpretation(
        summary_df
    )

    document_df = (
        calculate_document_consistency(
            consistency_df
        )
    )

    CONSISTENCY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_file = (
        CONSISTENCY_DIR
        / "consistency_dataset.csv"
    )

    pairwise_file = (
        CONSISTENCY_DIR
        / "pairwise_consistency.csv"
    )

    summary_file = (
        CONSISTENCY_DIR
        / "strategy_consistency.csv"
    )

    document_file = (
        CONSISTENCY_DIR
        / "document_consistency.csv"
    )

    consistency_df.to_csv(
        dataset_file,
        index=False,
    )

    pairwise_df.to_csv(
        pairwise_file,
        index=False,
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    document_df.to_csv(
        document_file,
        index=False,
    )

    print("\n" + "=" * 60)
    print("PAIRWISE CONSISTENCY")
    print("=" * 60)

    print(
        pairwise_df[
            [
                "strategy",
                "comparison",
                "observations",
                "kappa",
                "agreement_rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\n" + "=" * 60)
    print("STRATEGY-LEVEL CONSISTENCY")
    print("=" * 60)

    print(
        summary_df[
            [
                "strategy",
                "strategy_label",
                "mean_kappa",
                "mean_agreement_rate",
                "min_kappa",
                "max_kappa",
                "kappa_interpretation",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\n" + "=" * 60)
    print("FILES SAVED")
    print("=" * 60)

    print(dataset_file)
    print(pairwise_file)
    print(summary_file)
    print(document_file)

    print("\n" + "=" * 60)
    print("CONSISTENCY ANALYSIS COMPLETE")
    print("=" * 60)

    return (
        consistency_df,
        pairwise_df,
        summary_df,
        document_df,
    )


if __name__ == "__main__":
    run_consistency_analysis()