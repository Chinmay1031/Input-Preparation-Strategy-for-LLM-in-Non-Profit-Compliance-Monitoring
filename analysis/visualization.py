"""
Visualization module for dissertation evaluation results.

Reads derived CSV outputs from the analysis modules and generates
thesis-ready figures under results/figures/.

Run:
    python -m analysis.visualization
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

QUALITY_DIR = RESULTS_DIR / "quality"
DIMENSION_DIR = RESULTS_DIR / "dimension"
DOCUMENT_DIR = RESULTS_DIR / "document"
EFFICIENCY_DIR = RESULTS_DIR / "efficiency"
CONSISTENCY_DIR = RESULTS_DIR / "consistency"
FAITHFULNESS_DIR = RESULTS_DIR / "faithfulness"
STATISTICS_DIR = RESULTS_DIR / "statistics"
ERROR_DIR = RESULTS_DIR / "errors"


STRATEGY_ORDER = [
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
]


STRATEGY_LABELS = {
    "S1": "S1 Full Text",
    "S2": "S2 Section Filtering",
    "S3": "S3 Field Extraction",
    "S4": "S4 Hybrid",
    "S5": "S5 Extended Extraction",
}


DIMENSION_ORDER = [
    "revenue_concentration",
    "expense_spike",
    "pass_through_risk",
    "unallowable_expenditure",
    "audit_opinion",
    "going_concern",
]


DIMENSION_LABELS = {
    "revenue_concentration": "Revenue\nConcentration",
    "expense_spike": "Expense\nSpike",
    "pass_through_risk": "Pass-through\nRisk",
    "unallowable_expenditure": "Unallowable\nExpenditure",
    "audit_opinion": "Audit\nOpinion",
    "going_concern": "Going\nConcern",
}


def setup():
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def save_figure(fig, filename):
    """
    Save each figure as both PNG and PDF.

    PNG is convenient for inspection and presentations.
    PDF is suitable for inclusion in the dissertation.
    """

    png_path = FIGURES_DIR / f"{filename}.png"
    pdf_path = FIGURES_DIR / f"{filename}.pdf"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


def apply_strategy_order(df, column="strategy"):
    """
    Apply the fixed S1-S5 strategy order.

    Derived CSV files may contain:
    - short IDs such as S1
    - internal names such as S1_full
    - display labels

    These are normalised to S1-S5 before plotting.
    """

    df = df.copy()

    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
    )

    # Display labels used by some derived CSV files.
    reverse_labels = {
        label: strategy
        for strategy, label in STRATEGY_LABELS.items()
    }

    df[column] = df[column].replace(
        reverse_labels
    )

    # Internal strategy names used by the experiment pipeline.
    replacements = {
        "Full Text": "S1",
        "Section Filtering": "S2",
        "Field Extraction": "S3",
        "Hybrid": "S4",
        "Extended Extraction": "S5",
        "S1_full": "S1",
        "S2_sections": "S2",
        "S3_fields": "S3",
        "S4_hybrid": "S4",
        "S5_extended": "S5",
    }

    df[column] = df[column].replace(
        replacements
    )

    unexpected = sorted(
        set(df[column].dropna())
        - set(STRATEGY_ORDER)
    )

    if unexpected:
        raise ValueError(
            f"Unexpected strategy values in "
            f"'{column}': {unexpected}"
        )

    df[column] = pd.Categorical(
        df[column],
        categories=STRATEGY_ORDER,
        ordered=True,
    )

    return df.sort_values(column)


def plot_overall_f1():
    path = QUALITY_DIR / "overall_quality.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    df = pd.read_csv(path)

    df = apply_strategy_order(df)

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    bars = ax.bar(
        range(len(df)),
        df["f1"],
    )

    ax.set_xticks(
        range(len(df))
    )

    ax.set_xticklabels(
        [
            STRATEGY_LABELS[s]
            for s in df["strategy"]
        ],
        rotation=20,
        ha="right",
    )

    ax.set_ylabel(
        "Macro F1"
    )

    ax.set_xlabel(
        "Input Preparation Strategy"
    )

    ax.set_title(
        "Overall Compliance Reasoning Quality by Strategy"
    )

    ax.set_ylim(
        0,
        1.0,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        df["f1"],
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    save_figure(
        fig,
        "fig_overall_f1_by_strategy",
    )


def plot_dimension_f1():
    """
    Plot positive-class F1 by compliance dimension.

    Audit opinion is marked as N/A because the gold standard contains
    no positive cases for that dimension. A zero positive-class F1
    would therefore be misleading.
    """

    path = (
        DIMENSION_DIR
        / "dimension_strategy_metrics.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    df = pd.read_csv(path)

    required = [
        "strategy",
        "dimension",
        "f1",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing dimension columns: "
            + ", ".join(missing)
        )

    df = apply_strategy_order(df)

    df = df[
        df["dimension"].isin(
            DIMENSION_ORDER
        )
    ].copy()

    # Audit opinion has no positive gold-standard cases.
    # Therefore positive-class F1 is undefined.
    df.loc[
        df["dimension"] == "audit_opinion",
        "f1",
    ] = float("nan")

    pivot = df.pivot(
        index="dimension",
        columns="strategy",
        values="f1",
    )

    pivot = pivot.reindex(
        DIMENSION_ORDER
    )

    pivot = pivot.reindex(
        columns=STRATEGY_ORDER
    )

    fig, ax = plt.subplots(
        figsize=(11, 6.5)
    )

    x = range(
        len(DIMENSION_ORDER)
    )

    width = 0.15

    for i, strategy in enumerate(
        STRATEGY_ORDER
    ):

        values = pivot[
            strategy
        ].tolist()

        positions = [
            value + (i - 2) * width
            for value in x
        ]

        ax.bar(
            positions,
            values,
            width=width,
            label=STRATEGY_LABELS[
                strategy
            ],
        )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        [
            DIMENSION_LABELS[d]
            for d in DIMENSION_ORDER
        ]
    )

    ax.set_ylabel(
        "Positive-class F1",
        fontsize=12,
    )

    ax.set_xlabel(
        "Compliance Dimension",
        fontsize=12,
    )

    ax.set_title(
        "Compliance Reasoning Quality by Dimension and Strategy",
        fontsize=15,
        pad=10,
    )

    ax.set_ylim(
        0,
        1.1,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.13,
        ),
        ncol=3,
        frameon=False,
    )

    # Audit opinion has no positive gold-standard support.
    audit_index = DIMENSION_ORDER.index(
        "audit_opinion"
    )

    ax.text(
        audit_index,
        0.08,
        "N/A",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig_dimension_f1_by_strategy",
    )


def plot_tokens_vs_f1():
    """
    Plot average input tokens against overall F1.

    Strategy labels are shown next to each point.
    """

    efficiency_path = (
        EFFICIENCY_DIR
        / "strategy_efficiency.csv"
    )

    quality_path = (
        QUALITY_DIR
        / "overall_quality.csv"
    )

    if not efficiency_path.exists():
        raise FileNotFoundError(
            f"Missing file: {efficiency_path}"
        )

    if not quality_path.exists():
        raise FileNotFoundError(
            f"Missing file: {quality_path}"
        )

    efficiency = pd.read_csv(
        efficiency_path
    )

    quality = pd.read_csv(
        quality_path
    )

    df = efficiency.merge(
        quality[
            [
                "strategy",
                "f1",
            ]
        ],
        on="strategy",
        how="inner",
    )

    df = apply_strategy_order(df)

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        df["avg_input_tokens"],
        df["f1"],
        s=80,
    )

    for _, row in df.iterrows():

        ax.annotate(
            row["strategy"],
            (
                row["avg_input_tokens"],
                row["f1"],
            ),
            xytext=(
                7,
                7,
            ),
            textcoords="offset points",
            fontsize=10,
        )

    ax.set_xlabel(
        "Average Input Tokens"
    )

    ax.set_ylabel(
        "Macro F1"
    )

    ax.set_title(
        "Relationship Between Input Volume and Compliance Reasoning Quality"
    )

    ax.set_ylim(
        0.6,
        0.85,
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig_input_tokens_vs_f1",
    )


def plot_consistency():
    path = (
        CONSISTENCY_DIR
        / "strategy_consistency.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    df = pd.read_csv(path)

    df = apply_strategy_order(df)

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    bars = ax.bar(
        range(len(df)),
        df["mean_kappa"],
    )

    ax.set_xticks(
        range(len(df))
    )

    ax.set_xticklabels(
        [
            STRATEGY_LABELS[s]
            for s in df["strategy"]
        ],
        rotation=20,
        ha="right",
    )

    ax.set_ylabel(
        "Mean Cohen's κ"
    )

    ax.set_xlabel(
        "Input Preparation Strategy"
    )

    ax.set_title(
        "Output Consistency Across Repeated Runs"
    )

    ax.set_ylim(
        0,
        1.0,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        df["mean_kappa"],
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    save_figure(
        fig,
        "fig_consistency_kappa",
    )


def plot_faithfulness():
    """
    Plot evidence type composition.

    Shows:
    - Verbatim evidence
    - Synthesised evidence
    - Unsupported evidence
    """

    path = (
        FAITHFULNESS_DIR
        / "strategy_faithfulness.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    df = pd.read_csv(path)

    df = apply_strategy_order(df)

    required = [
        "strategy",
        "verbatim_rate",
        "synthesised_rate",
        "unsupported_rate",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing faithfulness columns: "
            + ", ".join(missing)
        )

    fig, ax = plt.subplots(
        figsize=(9.5, 6)
    )

    x = range(
        len(df)
    )

    ax.bar(
        x,
        df["verbatim_rate"] * 100,
        label="Verbatim",
    )

    ax.bar(
        x,
        df["synthesised_rate"] * 100,
        bottom=(
            df["verbatim_rate"]
            * 100
        ),
        label="Synthesised",
    )

    ax.bar(
        x,
        df["unsupported_rate"] * 100,
        bottom=(
            df["verbatim_rate"]
            + df["synthesised_rate"]
        ) * 100,
        label="Unsupported",
    )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        [
            STRATEGY_LABELS[s]
            for s in df["strategy"]
        ],
        rotation=20,
        ha="right",
    )

    ax.set_ylabel(
        "Evidence Cases (%)"
    )

    ax.set_xlabel(
        "Input Preparation Strategy"
    )

    ax.set_title(
        "Evidence Type in Compliance Findings"
    )

    ax.set_ylim(
        0,
        105,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.13,
        ),
        ncol=3,
        frameon=False,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig_faithfulness_evidence_type",
    )


def plot_document_f1():
    """
    Plot document-level F1 for all documents and strategies.

    Uses short publication-friendly document labels instead of
    internal document identifiers.
    """

    path = (
        DOCUMENT_DIR
        / "document_level_f1.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    df = pd.read_csv(path)

    required = [
        "gold_document_id",
        "strategy",
        "f1",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing document-level columns: "
            + ", ".join(missing)
        )

    df = apply_strategy_order(df)

    document_labels = {
        "AsianLawCaucus_AuditedFinancials_2025":
            "Asian Law\nCaucus\n(2025)",

        "BRAC_Liberia_AuditedFinancials_2023":
            "BRAC Liberia\n(2023)",

        "BRAC_Uganda_AuditedFinancials_2023":
            "BRAC Uganda\n(2023)",

        "CARE_USA_AuditedFinancials_2024":
            "CARE USA\n(2024)",

        "JusticeInAging_AuditedFinancials_2023":
            "Justice in Aging\n(2023)",

        "PATH_AnnualReport_2024":
            "PATH\n(2024)",

        "PublicCitizenFoundation_AuditedFinancials_2024":
            "Public Citizen\n(2024)",

        "RiverNetwork_AuditedFinancials_2023":
            "River Network\n(2023)",

        "RockingTheBoat_AuditedFinancials_2023":
            "Rocking the Boat\n(2023)",

        "SOMOSMayfair_AuditedFinancials_2024":
            "SOMOS Mayfair\n(2024)",

        "SaveTheChildrenFederation_AuditedFinancials_2024":
            "Save the Children\n(2024)",

        "WaterOrg_AuditedFinancials_2024":
            "Water.org\n(2024)",
    }

    missing_labels = [
        doc_id
        for doc_id in df[
            "gold_document_id"
        ].unique()
        if doc_id not in document_labels
    ]

    if missing_labels:
        raise ValueError(
            "Missing document labels for: "
            + ", ".join(missing_labels)
        )

    document_order = [
        "AsianLawCaucus_AuditedFinancials_2025",
        "BRAC_Liberia_AuditedFinancials_2023",
        "BRAC_Uganda_AuditedFinancials_2023",
        "CARE_USA_AuditedFinancials_2024",
        "JusticeInAging_AuditedFinancials_2023",
        "PATH_AnnualReport_2024",
        "PublicCitizenFoundation_AuditedFinancials_2024",
        "RiverNetwork_AuditedFinancials_2023",
        "RockingTheBoat_AuditedFinancials_2023",
        "SOMOSMayfair_AuditedFinancials_2024",
        "SaveTheChildrenFederation_AuditedFinancials_2024",
        "WaterOrg_AuditedFinancials_2024",
    ]

    pivot = df.pivot(
        index="gold_document_id",
        columns="strategy",
        values="f1",
    )

    pivot = pivot.reindex(
        document_order
    )

    pivot = pivot.reindex(
        columns=STRATEGY_ORDER
    )

    fig, ax = plt.subplots(
        figsize=(16, 10)
    )

    x = range(
        len(pivot.index)
    )

    width = 0.15

    for i, strategy in enumerate(
        STRATEGY_ORDER
    ):

        positions = [
            value + (i - 2) * width
            for value in x
        ]

        ax.bar(
            positions,
            pivot[strategy],
            width=width,
            label=STRATEGY_LABELS[
                strategy
            ],
        )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        [
            document_labels[doc_id]
            for doc_id in pivot.index
        ],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=11,
    )

    ax.set_ylabel(
        "Document-level F1",
        fontsize=13,
    )

    ax.set_xlabel(
        "Document",
        fontsize=13,
        labelpad=12,
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_title(
        "Variation in Compliance Reasoning Quality Across Documents",
        fontsize=17,
        pad=15,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.32,
        ),
        ncol=5,
        frameon=False,
        fontsize=11,
    )

    # Extra space for rotated labels and legend
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.92,
        bottom=0.42,
    )

    save_figure(
        fig,
        "fig_document_f1_distribution",
    )


def plot_quality_efficiency_tradeoff():
    path = (
        STATISTICS_DIR
        / "quality_efficiency_tradeoff.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    df = pd.read_csv(path)

    required = [
        "strategy",
        "input_token_reduction_pct",
        "f1",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing quality-efficiency columns: "
            + ", ".join(missing)
        )

    df = apply_strategy_order(df)

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        df["input_token_reduction_pct"],
        df["f1"],
        s=90,
    )

    for _, row in df.iterrows():

        ax.annotate(
            row["strategy"],
            (
                row["input_token_reduction_pct"],
                row["f1"],
            ),
            xytext=(
                7,
                7,
            ),
            textcoords="offset points",
            fontsize=10,
        )

    ax.set_xlabel(
        "Input Token Reduction (%)"
    )

    ax.set_ylabel(
        "Macro F1"
    )

    ax.set_title(
        "Quality–Efficiency Trade-off Across Input Strategies"
    )

    ax.set_xlim(
        -5,
        100,
    )

    ax.set_ylim(
        0.6,
        0.85,
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig_quality_efficiency_tradeoff",
    )


def main():

    print("=" * 70)
    print("DISSERTATION VISUALIZATION PIPELINE")
    print("=" * 70)

    setup()

    print("\nGenerating figures...\n")

    plot_overall_f1()
    plot_dimension_f1()
    plot_tokens_vs_f1()
    plot_consistency()
    plot_faithfulness()
    plot_document_f1()
    plot_quality_efficiency_tradeoff()

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)

    print("\nFigures saved to:")
    print(FIGURES_DIR)

    print("\nGenerated figures:")

    for path in sorted(
        FIGURES_DIR.glob("*.png")
    ):
        print(
            f"  - {path.name}"
        )


if __name__ == "__main__":
    main()