"""
results_table.py
----------------
Generates the master results table — Table 1 in the thesis.
Combines all four metric dimensions into one pandas DataFrame.
"""

import pandas as pd
from typing import List, Dict
from src.output_processing.result_schema import ExperimentResult
from .quality_scorer      import compute_quality_metrics
from .efficiency_scorer   import compute_efficiency_metrics
from .consistency_scorer  import compute_consistency_metrics
from .faithfulness_scorer import compute_faithfulness_metrics


STRATEGY_ORDER = ["S1_full", "S2_sections", "S3_fields", "S4_hybrid"]

STRATEGY_LABELS = {
    "S1_full":     "S1 — Full text",
    "S2_sections": "S2 — Section filter",
    "S3_fields":   "S3 — Field extraction",
    "S4_hybrid":   "S4 — Hybrid",
}


def generate_master_table(
    results: List[ExperimentResult],
    gold_standard_path: str = "data/gold_standard/gold_standard.csv"
) -> pd.DataFrame:
    """
    Generate the master results table combining all four
    evaluation dimensions.

    Args:
        results:            all ExperimentResult objects
        gold_standard_path: path to gold standard CSV

    Returns:
        DataFrame with one row per strategy — thesis Table 1
    """
    quality     = compute_quality_metrics(results, gold_standard_path)
    efficiency  = compute_efficiency_metrics(results)
    consistency = compute_consistency_metrics(results)
    faithfulness = compute_faithfulness_metrics(results)

    rows = []
    for strategy in STRATEGY_ORDER:
        if strategy not in quality:
            continue

        q = quality.get(strategy, {})
        e = efficiency.get(strategy, {})
        c = consistency.get(strategy, {})
        f = faithfulness.get(strategy, {})

        rows.append({
            "Strategy":          STRATEGY_LABELS.get(strategy, strategy),
            "Precision":         q.get("precision", "-"),
            "Recall":            q.get("recall", "-"),
            "F1":                q.get("f1", "-"),
            "Avg tokens":        e.get("avg_input_tokens", "-"),
            "Cost/doc ($)":      e.get("cost_per_doc_usd", "-"),
            "Token reduction":   e.get("token_reduction_vs_s1", "-"),
            "Kappa":             c.get("kappa", "-"),
            "Consistency":       c.get("kappa_interpretation", "-"),
            "Faithfulness":      f.get("avg_faithfulness", "-"),
            "Hallucination %":   f.get("hallucination_pct", "-"),
        })

    return pd.DataFrame(rows)


def print_master_table(df: pd.DataFrame) -> None:
    """Print the master results table to terminal."""
    print("\n" + "="*90)
    print("MASTER RESULTS TABLE — Thesis Table 1")
    print("="*90)
    print(df.to_string(index=False))
    print("="*90)