"""
Statistical analysis for the LLM input-strategy experiment.

Analyses:
    1. Input token volume vs. compliance reasoning quality.
    2. Monotonicity of the token-volume / F1 relationship.
    3. Pairwise strategy changes.
    4. Quality-efficiency trade-offs.

The analysis is descriptive because only five strategy-level
observations are available.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"

QUALITY_DIR = RESULTS_DIR / "quality"
EFFICIENCY_DIR = RESULTS_DIR / "efficiency"
STATISTICS_DIR = RESULTS_DIR / "statistics"

QUALITY_FILE = QUALITY_DIR / "overall_quality.csv"
EFFICIENCY_FILE = EFFICIENCY_DIR / "strategy_efficiency.csv"

STRATEGY_ORDER = [
    "S1_full",
    "S2_sections",
    "S3_fields",
    "S4_hybrid",
    "S5_extended",
]

STRATEGY_LABELS = {
    "S1_full": "S1 – Full text",
    "S2_sections": "S2 – Section filtering",
    "S3_fields": "S3 – Field extraction",
    "S4_hybrid": "S4 – Hybrid",
    "S5_extended": "S5 – Extended extraction",
}

def load_analysis_data():

    if not QUALITY_FILE.exists():
        raise FileNotFoundError(
            f"Quality results not found:\n{QUALITY_FILE}\n\n"
            "Run analysis.quality_analysis first."
        )

    if not EFFICIENCY_FILE.exists():
        raise FileNotFoundError(
            f"Efficiency results not found:\n{EFFICIENCY_FILE}\n\n"
            "Run analysis.efficiency_analysis first."
        )

    quality = pd.read_csv(
        QUALITY_FILE
    )

    efficiency = pd.read_csv(
        EFFICIENCY_FILE
    )

    print(
        f"Quality records loaded: {len(quality)}"
    )

    print(
        f"Efficiency records loaded: {len(efficiency)}"
    )

    return quality, efficiency

def normalise_efficiency_columns(
    efficiency
):
    """Normalise the efficiency output to the column names the rest of this module expects."""

    efficiency = efficiency.copy()

    print("\nEfficiency columns detected:")
    print(
        ", ".join(
            efficiency.columns.tolist()
        )
    )

    aliases = {

        "avg_input_tokens": [
            "avg_input_tokens",
            "average_input_tokens",
            "input_tokens",
            "avg_tokens_input",
        ],

        "avg_output_tokens": [
            "avg_output_tokens",
            "average_output_tokens",
            "output_tokens",
            "avg_tokens_output",
        ],

        "avg_total_tokens": [
            "avg_total_tokens",
            "average_total_tokens",
            "total_tokens",
            "avg_tokens_total",
        ],

        "cost_per_document": [
            "cost_per_document",
            "avg_cost_per_document",
            "cost_per_doc",
            "average_cost_per_document",
            "avg_cost",
            "cost_usd_per_document",
        ],

        "input_token_reduction_pct": [
            "input_token_reduction_pct",
            "input_reduction_pct",
            "token_reduction_pct",
            "input_token_reduction",
        ],

        "cost_reduction_pct": [
            "cost_reduction_pct",
            "cost_reduction",
        ],
    }

    rename_map = {}

    for standard_name, candidates in aliases.items():

        if standard_name in efficiency.columns:
            continue

        for candidate in candidates:

            if candidate in efficiency.columns:
                rename_map[candidate] = standard_name
                break

    if rename_map:
        efficiency = efficiency.rename(
            columns=rename_map
        )

    return efficiency

def prepare_strategy_dataset(
    quality,
    efficiency,
):

    quality = quality.copy()

    efficiency = (
        normalise_efficiency_columns(
            efficiency
        )
    )

    quality["strategy"] = (
        quality["strategy"].astype(str)
    )

    efficiency["strategy"] = (
        efficiency["strategy"].astype(str)
    )

    required_quality = [
        "strategy",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "judgements",
        "correct",
    ]

    missing_quality = [
        column
        for column in required_quality
        if column not in quality.columns
    ]

    if missing_quality:
        raise ValueError(
            "Missing quality columns: "
            + ", ".join(missing_quality)
        )

    required_efficiency = [
        "strategy",
        "avg_input_tokens",
    ]

    missing_efficiency = [
        column
        for column in required_efficiency
        if column not in efficiency.columns
    ]

    if missing_efficiency:
        raise ValueError(
            "Missing required efficiency columns: "
            + ", ".join(missing_efficiency)
            + "\n\nAvailable columns: "
            + ", ".join(efficiency.columns)
        )

    optional_efficiency = [
        "avg_output_tokens",
        "avg_total_tokens",
        "cost_per_document",
        "input_token_reduction_pct",
        "cost_reduction_pct",
    ]

    quality_selected = quality[
        required_quality
    ].copy()

    efficiency_columns = [
        column
        for column in (
            required_efficiency
            + optional_efficiency
        )
        if column in efficiency.columns
    ]

    efficiency_selected = efficiency[
        efficiency_columns
    ].copy()

    merged = pd.merge(
        quality_selected,
        efficiency_selected,
        on="strategy",
        how="inner",
    )

    merged["strategy_order"] = (
        merged["strategy"]
        .map(
            {
                strategy: index
                for index, strategy
                in enumerate(STRATEGY_ORDER)
            }
        )
    )

    merged = (
        merged
        .sort_values(
            "strategy_order"
        )
        .drop(
            columns=["strategy_order"]
        )
        .reset_index(drop=True)
    )

    merged["strategy_label"] = (
        merged["strategy"]
        .map(STRATEGY_LABELS)
    )

    if len(merged) != 5:
        raise ValueError(
            "Expected five strategy-level observations "
            f"but found {len(merged)}."
        )

    return merged

def calculate_correlations(
    strategy_df
):

    x = (
        strategy_df[
            "avg_input_tokens"
        ]
        .astype(float)
    )

    y = (
        strategy_df[
            "f1"
        ]
        .astype(float)
    )

    pearson_r = x.corr(
        y,
        method="pearson",
    )

    spearman_rho = x.corr(
        y,
        method="spearman",
    )

    return {
        "n_strategies": len(
            strategy_df
        ),
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
    }

def check_monotonicity(
    strategy_df
):

    ordered = (
        strategy_df
        .sort_values(
            "avg_input_tokens"
        )
        .reset_index(drop=True)
    )

    tokens = (
        ordered[
            "avg_input_tokens"
        ]
        .to_numpy()
    )

    f1 = (
        ordered[
            "f1"
        ]
        .to_numpy()
    )

    f1_differences = np.diff(
        f1
    )

    increasing_f1 = np.all(
        f1_differences >= 0
    )

    decreasing_f1 = np.all(
        f1_differences <= 0
    )

    strictly_increasing = np.all(
        f1_differences > 0
    )

    strictly_decreasing = np.all(
        f1_differences < 0
    )

    directions = np.sign(
        f1_differences
    )

    non_zero_directions = (
        directions[
            directions != 0
        ]
    )

    direction_changes = 0

    if len(
        non_zero_directions
    ) > 1:

        direction_changes = int(
            np.sum(
                non_zero_directions[1:]
                !=
                non_zero_directions[:-1]
            )
        )

    if strictly_increasing:
        relationship = (
            "Strictly increasing"
        )

    elif strictly_decreasing:
        relationship = (
            "Strictly decreasing"
        )

    elif increasing_f1:
        relationship = (
            "Non-decreasing"
        )

    elif decreasing_f1:
        relationship = (
            "Non-increasing"
        )

    else:
        relationship = (
            "Non-monotonic"
        )

    return {
        "relationship": relationship,
        "direction_changes": direction_changes,
    }, ordered

def calculate_pairwise_changes(
    strategy_df
):

    rows = []

    for i in range(
        len(strategy_df) - 1
    ):

        current = strategy_df.iloc[i]
        following = strategy_df.iloc[i + 1]

        current_tokens = float(
            current[
                "avg_input_tokens"
            ]
        )

        following_tokens = float(
            following[
                "avg_input_tokens"
            ]
        )

        current_f1 = float(
            current["f1"]
        )

        following_f1 = float(
            following["f1"]
        )

        token_change = (
            following_tokens
            - current_tokens
        )

        f1_change = (
            following_f1
            - current_f1
        )

        token_change_pct = (
            token_change
            / current_tokens
            * 100
            if current_tokens != 0
            else np.nan
        )

        f1_change_pct = (
            f1_change
            / current_f1
            * 100
            if current_f1 != 0
            else np.nan
        )

        rows.append({
            "from_strategy": current[
                "strategy"
            ],

            "to_strategy": following[
                "strategy"
            ],

            "from_strategy_label": current[
                "strategy_label"
            ],

            "to_strategy_label": following[
                "strategy_label"
            ],

            "from_input_tokens": current_tokens,

            "to_input_tokens": following_tokens,

            "token_change": token_change,

            "token_change_pct": token_change_pct,

            "from_f1": current_f1,

            "to_f1": following_f1,

            "f1_change": f1_change,

            "f1_change_pct": f1_change_pct,
        })

    return pd.DataFrame(
        rows
    )

def calculate_efficiency_frontier(
    strategy_df
):

    columns = [
        "strategy",
        "strategy_label",
        "avg_input_tokens",
        "f1",
    ]

    optional_columns = [
        "cost_per_document",
        "input_token_reduction_pct",
        "cost_reduction_pct",
    ]

    for column in optional_columns:

        if column in strategy_df.columns:
            columns.append(
                column
            )

    result = strategy_df[
        columns
    ].copy()

    result[
        "f1_per_1000_input_tokens"
    ] = (
        result["f1"]
        /
        (
            result[
                "avg_input_tokens"
            ]
            / 1000
        )
    )

    return result

def run_statistical_analysis():

    print("=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    quality, efficiency = (
        load_analysis_data()
    )

    strategy_df = (
        prepare_strategy_dataset(
            quality,
            efficiency,
        )
    )

    print("\n" + "=" * 60)
    print("STRATEGY-LEVEL ANALYSIS DATA")
    print("=" * 60)

    display_columns = [
        "strategy",
        "strategy_label",
        "f1",
        "avg_input_tokens",
    ]

    for column in [
        "cost_per_document",
        "input_token_reduction_pct",
    ]:
        if column in strategy_df.columns:
            display_columns.append(
                column
            )

    print(
        strategy_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    correlations = (
        calculate_correlations(
            strategy_df
        )
    )

    print("\n" + "=" * 60)
    print("TOKEN VOLUME VS. COMPLIANCE QUALITY")
    print("=" * 60)

    print(
        f"Number of strategy observations: "
        f"{correlations['n_strategies']}"
    )

    print(
        f"Pearson correlation (r): "
        f"{correlations['pearson_r']:.3f}"
    )

    print(
        f"Spearman correlation (rho): "
        f"{correlations['spearman_rho']:.3f}"
    )

    print(
        "\nThese correlations are treated as "
        "descriptive because n = 5."
    )

    monotonicity, ordered_df = (
        check_monotonicity(
            strategy_df
        )
    )

    print("\n" + "=" * 60)
    print("MONOTONICITY CHECK")
    print("=" * 60)

    print(
        f"Relationship: "
        f"{monotonicity['relationship']}"
    )

    print(
        f"Direction changes in F1: "
        f"{monotonicity['direction_changes']}"
    )

    print(
        "\nStrategies ordered by input-token volume:"
    )

    print(
        ordered_df[
            [
                "strategy",
                "avg_input_tokens",
                "f1",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    pairwise_df = (
        calculate_pairwise_changes(
            strategy_df
        )
    )

    print("\n" + "=" * 60)
    print("PAIRWISE STRATEGY CHANGES")
    print("=" * 60)

    print(
        pairwise_df[
            [
                "from_strategy",
                "to_strategy",
                "token_change_pct",
                "f1_change",
                "f1_change_pct",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    frontier_df = (
        calculate_efficiency_frontier(
            strategy_df
        )
    )

    print("\n" + "=" * 60)
    print("QUALITY-EFFICIENCY TRADE-OFF")
    print("=" * 60)

    print(
        frontier_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    STATISTICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    strategy_output = (
        STATISTICS_DIR
        / "strategy_statistical_dataset.csv"
    )

    ordered_output = (
        STATISTICS_DIR
        / "token_quality_ordered.csv"
    )

    pairwise_output = (
        STATISTICS_DIR
        / "pairwise_strategy_changes.csv"
    )

    frontier_output = (
        STATISTICS_DIR
        / "quality_efficiency_tradeoff.csv"
    )

    correlation_output = (
        STATISTICS_DIR
        / "token_quality_correlations.csv"
    )

    strategy_df.to_csv(
        strategy_output,
        index=False,
    )

    ordered_df.to_csv(
        ordered_output,
        index=False,
    )

    pairwise_df.to_csv(
        pairwise_output,
        index=False,
    )

    frontier_df.to_csv(
        frontier_output,
        index=False,
    )

    pd.DataFrame([
        correlations
    ]).to_csv(
        correlation_output,
        index=False,
    )

    print("\n" + "=" * 60)
    print("FILES SAVED")
    print("=" * 60)

    print(strategy_output)
    print(ordered_output)
    print(pairwise_output)
    print(frontier_output)
    print(correlation_output)

    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("=" * 60)

    return {
        "strategy": strategy_df,
        "ordered": ordered_df,
        "pairwise": pairwise_df,
        "frontier": frontier_df,
        "correlations": correlations,
    }

if __name__ == "__main__":
    run_statistical_analysis()