import pandas as pd

from analysis.config import (
    STRATEGIES,
    STRATEGY_LABELS,
    EFFICIENCY_DIR,
)

from analysis.validation import validate_dataset


# Existing experiment cost assumptions
COST_PER_1K_INPUT_TOKENS = 0.005
COST_PER_1K_OUTPUT_TOKENS = 0.015

# Full-text strategy is the efficiency baseline
BASELINE_STRATEGY = "S1_full"

def build_efficiency_dataset(records):
    """
    Build the efficiency dataset from the raw experiment records.

    Efficiency is calculated across all three runs for each
    strategy. This preserves the experimental token measurements
    rather than recalculating tokens from source documents.
    """

    rows = []

    for record in records:

        strategy = record["strategy"]

        input_tokens = record.get("tokens_input")
        output_tokens = record.get("tokens_output")

        if input_tokens is None:
            raise ValueError(
                f"Missing input_tokens for "
                f"{record['doc_id']} / {strategy} / "
                f"run {record['run']}"
            )

        if output_tokens is None:
            raise ValueError(
                f"Missing output_tokens for "
                f"{record['doc_id']} / {strategy} / "
                f"run {record['run']}"
            )

        input_tokens = float(input_tokens)
        output_tokens = float(output_tokens)

        total_tokens = (
            input_tokens +
            output_tokens
        )

        input_cost = (
            input_tokens / 1000
        ) * COST_PER_1K_INPUT_TOKENS

        output_cost = (
            output_tokens / 1000
        ) * COST_PER_1K_OUTPUT_TOKENS

        total_cost = (
            input_cost +
            output_cost
        )

        rows.append({
            "doc_id": record["doc_id"],
            "gold_document_id": record.get(
                "gold_document_id"
            ),
            "strategy": strategy,
            "strategy_label": STRATEGY_LABELS[strategy],
            "run": record["run"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
        })

    efficiency_df = pd.DataFrame(rows)

    if efficiency_df.empty:
        raise ValueError(
            "Efficiency dataset is empty."
        )

    return efficiency_df


def calculate_strategy_efficiency(efficiency_df):
    """
    Calculate mean token usage and cost per strategy.

    The existing experiment scorer uses average input tokens
    across Run 0 for its headline efficiency figures.

    We therefore report both:

    1. Run-0 input-token metrics, matching the established
       experiment results.
    2. All-run averages as an additional descriptive statistic.
    """

    rows = []

    for strategy in STRATEGIES:

        all_runs = efficiency_df[
            efficiency_df["strategy"] == strategy
        ]

        run_zero = all_runs[
            all_runs["run"] == 0
        ]

        if run_zero.empty:
            raise ValueError(
                f"No Run 0 records found for {strategy}."
            )

        avg_input_tokens = (
            run_zero["input_tokens"].mean()
        )

        avg_output_tokens = (
            run_zero["output_tokens"].mean()
        )

        avg_total_tokens = (
            run_zero["total_tokens"].mean()
        )

        avg_input_cost = (
            run_zero["input_cost"].mean()
        )

        avg_output_cost = (
            run_zero["output_cost"].mean()
        )

        avg_total_cost = (
            run_zero["total_cost"].mean()
        )

        all_run_input = (
            all_runs["input_tokens"].mean()
        )

        all_run_output = (
            all_runs["output_tokens"].mean()
        )

        all_run_total = (
            all_runs["total_tokens"].mean()
        )

        all_run_cost = (
            all_runs["total_cost"].mean()
        )

        rows.append({
            "strategy": strategy,
            "strategy_label": STRATEGY_LABELS[strategy],
            "documents": run_zero["doc_id"].nunique(),
            "runs": all_runs["run"].nunique(),

            "avg_input_tokens": avg_input_tokens,
            "avg_output_tokens": avg_output_tokens,
            "avg_total_tokens": avg_total_tokens,
            "avg_input_cost": avg_input_cost,
            "avg_output_cost": avg_output_cost,
            "avg_total_cost": avg_total_cost,

            "all_run_avg_input_tokens": all_run_input,
            "all_run_avg_output_tokens": all_run_output,
            "all_run_avg_total_tokens": all_run_total,
            "all_run_avg_total_cost": all_run_cost,
        })

    summary_df = pd.DataFrame(rows)

    return summary_df


def calculate_baseline_reductions(summary_df):
    """
    Calculate token and cost reductions relative to S1,
    the full-text baseline.
    """

    baseline = summary_df[
        summary_df["strategy"] == BASELINE_STRATEGY
    ]

    if baseline.empty:
        raise ValueError(
            f"Baseline strategy "
            f"{BASELINE_STRATEGY} not found."
        )

    baseline_input = float(
        baseline.iloc[0]["avg_input_tokens"]
    )

    baseline_total_cost = float(
        baseline.iloc[0]["avg_total_cost"]
    )

    result = summary_df.copy()

    result["input_token_reduction_pct"] = (
        1 -
        result["avg_input_tokens"] /
        baseline_input
    ) * 100

    result["cost_reduction_pct"] = (
        1 -
        result["avg_total_cost"] /
        baseline_total_cost
    ) * 100

    # Avoid tiny floating-point values such as
    # -0.0000000001 for the baseline.
    result["input_token_reduction_pct"] = (
        result["input_token_reduction_pct"]
        .clip(lower=0)
    )

    result["cost_reduction_pct"] = (
        result["cost_reduction_pct"]
        .clip(lower=0)
    )

    return result


def calculate_document_efficiency(efficiency_df):
    """
    Calculate Run-0 token usage and cost for every
    document × strategy combination.

    This dataset can later be used to examine variation
    in efficiency across documents.
    """

    run_zero = efficiency_df[
        efficiency_df["run"] == 0
    ].copy()

    document_df = (
        run_zero
        .groupby(
            [
                "gold_document_id",
                "strategy",
                "strategy_label",
            ],
            as_index=False,
        )
        .agg(
            input_tokens=("input_tokens", "mean"),
            output_tokens=("output_tokens", "mean"),
            total_tokens=("total_tokens", "mean"),
            total_cost=("total_cost", "mean"),
        )
    )

    return document_df


def run_efficiency_analysis():

    records, _ = validate_dataset()

    print("\n" + "=" * 60)
    print("EFFICIENCY ANALYSIS")
    print("=" * 60)

    efficiency_df = build_efficiency_dataset(
        records
    )

    print("\nEFFICIENCY DATA")
    print("-" * 60)

    print(
        f"Records: {len(efficiency_df)}"
    )

    print(
        f"Documents: "
        f"{efficiency_df['doc_id'].nunique()}"
    )

    print(
        f"Strategies: "
        f"{efficiency_df['strategy'].nunique()}"
    )

    print(
        f"Runs: "
        f"{efficiency_df['run'].nunique()}"
    )

    summary_df = calculate_strategy_efficiency(
        efficiency_df
    )

    summary_df = calculate_baseline_reductions(
        summary_df
    )

    document_df = calculate_document_efficiency(
        efficiency_df
    )

    EFFICIENCY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_file = (
        EFFICIENCY_DIR
        / "efficiency_dataset.csv"
    )

    summary_file = (
        EFFICIENCY_DIR
        / "strategy_efficiency.csv"
    )

    document_file = (
        EFFICIENCY_DIR
        / "document_efficiency.csv"
    )

    efficiency_df.to_csv(
        raw_file,
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
    print("STRATEGY-LEVEL EFFICIENCY")
    print("=" * 60)

    display_columns = [
        "strategy",
        "strategy_label",
        "avg_input_tokens",
        "avg_output_tokens",
        "avg_total_tokens",
        "avg_total_cost",
        "input_token_reduction_pct",
        "cost_reduction_pct",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\n" + "=" * 60)
    print("HEADLINE INPUT-TOKEN METRICS")
    print("=" * 60)

    for _, row in summary_df.iterrows():

        print(
            f"{row['strategy']}: "
            f"{row['avg_input_tokens']:,.0f} input tokens | "
            f"${row['avg_total_cost']:.5f}/document | "
            f"{row['input_token_reduction_pct']:.1f}% reduction"
        )

    print("\n" + "=" * 60)
    print("FILES SAVED")
    print("=" * 60)

    print(raw_file)
    print(summary_file)
    print(document_file)

    print("\n" + "=" * 60)
    print("EFFICIENCY ANALYSIS COMPLETE")
    print("=" * 60)

    return (
        efficiency_df,
        summary_df,
        document_df,
    )


if __name__ == "__main__":
    run_efficiency_analysis()