

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


# ---------------------------------------------------------------------------
# Publication-quality plotting helpers
# ---------------------------------------------------------------------------

import numpy as np
from matplotlib.ticker import PercentFormatter


def set_publication_style():
    """Apply one consistent, dissertation-friendly visual style."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 14,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "axes.labelweight": "medium",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "xtick.major.size": 0,
        "ytick.major.size": 3,
    })


def _strategy_short_labels():
    return ["S1\nFull Text", "S2\nSection Filtering", "S3\nField Extraction",
            "S4\nHybrid", "S5\nExtended Extraction"]


def _strategy_colors():
    # Uses Matplotlib's default cycle; no hard-coded colour choices.
    return plt.rcParams["axes.prop_cycle"].by_key()["color"][:5]


def _add_value_labels(ax, bars, values, fmt="{:.3f}", dy=0.012):
    for bar, value in zip(bars, values):
        if pd.isna(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + dy,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _style_axes(ax):
    ax.tick_params(axis="both", length=0)
    ax.spines["left"].set_alpha(0.45)
    ax.spines["bottom"].set_alpha(0.45)


def plot_overall_f1():
    path = QUALITY_DIR / "overall_quality.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = apply_strategy_order(pd.read_csv(path))
    values = df["f1"].to_numpy()

    fig, ax = plt.subplots(figsize=(9.5, 5.8), layout="constrained")
    x = np.arange(len(df))
    bars = ax.bar(x, values, width=0.62)

    ax.set_xticks(x, _strategy_short_labels())
    ax.set_ylabel("Macro F1")
    ax.set_xlabel("Input preparation strategy", labelpad=10)
    ax.set_title("Overall Compliance Reasoning Quality", pad=14)
    ax.set_ylim(0, 0.9)
    ax.set_yticks(np.arange(0, 0.91, 0.1))
    _add_value_labels(ax, bars, values, dy=0.018)
    _style_axes(ax)

    save_figure(fig, "fig_overall_f1_by_strategy")


def plot_dimension_f1():
    path = DIMENSION_DIR / "dimension_strategy_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    required = ["strategy", "dimension", "f1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing dimension columns: " + ", ".join(missing))

    df = apply_strategy_order(df)
    df = df[df["dimension"].isin(DIMENSION_ORDER)].copy()
    df.loc[df["dimension"] == "audit_opinion", "f1"] = np.nan

    pivot = (
        df.pivot(index="dimension", columns="strategy", values="f1")
        .reindex(index=DIMENSION_ORDER, columns=STRATEGY_ORDER)
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.3), layout="constrained")
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")

    ax.set_xticks(np.arange(len(STRATEGY_ORDER)), _strategy_short_labels())
    ax.set_yticks(
        np.arange(len(DIMENSION_ORDER)),
        [DIMENSION_LABELS[d].replace("\n", " ") for d in DIMENSION_ORDER],
    )
    ax.set_xlabel("Input preparation strategy", labelpad=10)
    ax.set_ylabel("Compliance dimension", labelpad=10)
    ax.set_title("Compliance Reasoning Quality by Dimension", pad=14)

    values = pivot.to_numpy(dtype=float)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            text = "N/A" if np.isnan(value) else f"{value:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)

    # Keep the colour scale explicit and comparable across strategies.
    image.set_clim(0, 1)
    cbar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_label("Positive-class F1")
    cbar.ax.tick_params(length=0)
    _style_axes(ax)

    save_figure(fig, "fig_dimension_f1_by_strategy")


def plot_tokens_vs_f1():
    efficiency_path = EFFICIENCY_DIR / "strategy_efficiency.csv"
    quality_path = QUALITY_DIR / "overall_quality.csv"
    if not efficiency_path.exists():
        raise FileNotFoundError(f"Missing file: {efficiency_path}")
    if not quality_path.exists():
        raise FileNotFoundError(f"Missing file: {quality_path}")

    efficiency = pd.read_csv(efficiency_path)
    quality = pd.read_csv(quality_path)
    df = apply_strategy_order(
        efficiency.merge(quality[["strategy", "f1"]], on="strategy", how="inner")
    )

    fig, ax = plt.subplots(figsize=(9.5, 6.2), layout="constrained")
    colors = _strategy_colors()

    for i, strategy in enumerate(STRATEGY_ORDER):
        row = df[df["strategy"] == strategy].iloc[0]
        ax.scatter(
            row["avg_input_tokens"], row["f1"],
            s=125, marker="o", color=colors[i],
            edgecolor="white", linewidth=1.2,
            zorder=3, label=strategy,
        )
        ax.annotate(
            strategy,
            (row["avg_input_tokens"], row["f1"]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=9.5, fontweight="semibold",
        )

    ax.set_xlabel("Average input tokens")
    ax.set_ylabel("Macro F1")
    ax.set_title("Input Volume and Compliance Reasoning Quality", pad=14)
    ax.set_ylim(0.60, 0.85)
    ax.legend(title="Strategy", ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), frameon=False)
    _style_axes(ax)

    save_figure(fig, "fig_input_tokens_vs_f1")


def plot_consistency():
    path = CONSISTENCY_DIR / "strategy_consistency.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = apply_strategy_order(pd.read_csv(path))
    values = df["mean_kappa"].to_numpy()

    fig, ax = plt.subplots(figsize=(9.5, 5.8), layout="constrained")
    x = np.arange(len(df))

    # Lollipop presentation keeps the five values visually separated.
    ax.vlines(x, 0, values, linewidth=2.2, alpha=0.55)
    ax.scatter(x, values, s=115, zorder=3)

    for xi, value in zip(x, values):
        ax.text(xi, value + 0.025, f"{value:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x, _strategy_short_labels())
    ax.set_ylabel("Mean Cohen's $\\kappa$")
    ax.set_xlabel("Input preparation strategy", labelpad=10)
    ax.set_title("Output Consistency Across Repeated Runs", pad=14)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    _style_axes(ax)

    save_figure(fig, "fig_consistency_kappa")


def plot_faithfulness():
    path = FAITHFULNESS_DIR / "strategy_faithfulness.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = apply_strategy_order(pd.read_csv(path))
    required = ["strategy", "verbatim_rate", "synthesised_rate", "unsupported_rate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing faithfulness columns: " + ", ".join(missing))

    fig, ax = plt.subplots(figsize=(9.8, 6.0), layout="constrained")
    x = np.arange(len(df))
    width = 0.66

    verbatim = df["verbatim_rate"].to_numpy() * 100
    synthesised = df["synthesised_rate"].to_numpy() * 100
    unsupported = df["unsupported_rate"].to_numpy() * 100

    ax.bar(x, verbatim, width, label="Verbatim")
    ax.bar(x, synthesised, width, bottom=verbatim, label="Synthesised")
    ax.bar(x, unsupported, width, bottom=verbatim + synthesised, label="Unsupported")

    ax.set_xticks(x, _strategy_short_labels())
    ax.set_ylabel("Evidence cases (%)")
    ax.set_xlabel("Input preparation strategy", labelpad=10)
    ax.set_title("Evidence Type in Compliance Findings", pad=14)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    _style_axes(ax)

    save_figure(fig, "fig_faithfulness_evidence_type")


def plot_document_f1():
    path = DOCUMENT_DIR / "document_level_f1.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    required = ["gold_document_id", "strategy", "f1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing document-level columns: " + ", ".join(missing))

    df = apply_strategy_order(df)

    document_labels = {
        "AsianLawCaucus_AuditedFinancials_2025": "Asian Law Caucus",
        "BRAC_Liberia_AuditedFinancials_2023": "BRAC Liberia",
        "BRAC_Uganda_AuditedFinancials_2023": "BRAC Uganda",
        "CARE_USA_AuditedFinancials_2024": "CARE USA",
        "JusticeInAging_AuditedFinancials_2023": "Justice in Aging",
        "PATH_AnnualReport_2024": "PATH",
        "PublicCitizenFoundation_AuditedFinancials_2024": "Public Citizen",
        "RiverNetwork_AuditedFinancials_2023": "River Network",
        "RockingTheBoat_AuditedFinancials_2023": "Rocking the Boat",
        "SOMOSMayfair_AuditedFinancials_2024": "SOMOS Mayfair",
        "SaveTheChildrenFederation_AuditedFinancials_2024": "Save the Children",
        "WaterOrg_AuditedFinancials_2024": "Water.org",
    }

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

    pivot = (
        df.pivot(index="gold_document_id", columns="strategy", values="f1")
        .reindex(index=document_order, columns=STRATEGY_ORDER)
    )

    fig, ax = plt.subplots(figsize=(10.8, 7.0), layout="constrained")
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")

    ax.set_xticks(np.arange(len(STRATEGY_ORDER)), [s for s in STRATEGY_ORDER])
    ax.set_yticks(np.arange(len(document_order)),
                  [document_labels[d] for d in document_order])
    ax.set_xlabel("Input preparation strategy", labelpad=10)
    ax.set_ylabel("Document", labelpad=10)
    ax.set_title("Variation in Compliance Reasoning Quality Across Documents", pad=14)

    values = pivot.to_numpy(dtype=float)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.isnan(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}",
                        ha="center", va="center", fontsize=8.5)

    image.set_clim(0, 1)
    cbar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.035)
    cbar.set_label("Document-level F1")
    cbar.ax.tick_params(length=0)
    _style_axes(ax)

    save_figure(fig, "fig_document_f1_distribution")


def plot_quality_efficiency_tradeoff():
    path = STATISTICS_DIR / "quality_efficiency_tradeoff.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = apply_strategy_order(pd.read_csv(path))
    required = ["strategy", "input_token_reduction_pct", "f1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing quality-efficiency columns: " + ", ".join(missing))

    fig, ax = plt.subplots(figsize=(9.5, 6.2), layout="constrained")
    colors = _strategy_colors()

    for i, strategy in enumerate(STRATEGY_ORDER):
        row = df[df["strategy"] == strategy].iloc[0]
        ax.scatter(
            row["input_token_reduction_pct"], row["f1"],
            s=140, color=colors[i], edgecolor="white", linewidth=1.2,
            zorder=3, label=strategy,
        )
        ax.annotate(
            strategy,
            (row["input_token_reduction_pct"], row["f1"]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=9.5, fontweight="semibold",
        )

    ax.set_xlabel("Input token reduction (%)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Quality–Efficiency Trade-off Across Input Strategies", pad=14)
    ax.set_xlim(-5, 100)
    ax.set_ylim(0.60, 0.85)
    ax.xaxis.set_major_formatter(PercentFormatter(100))
    ax.legend(title="Strategy", ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), frameon=False)
    _style_axes(ax)

    save_figure(fig, "fig_quality_efficiency_tradeoff")


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