"""
quality_scorer.py
-----------------
Computes Precision, Recall, and F1 score for each strategy
by comparing LLM verdicts against the gold standard.

Uses macro averaging — treats each compliance dimension equally
regardless of class imbalance.

Recall is emphasised in the discussion because in compliance
contexts, missing a real flag (false negative) is more serious
than raising a false alarm (false positive).
"""

from typing import List, Dict
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix
)
from src.output_processing.result_schema import ExperimentResult
from .gold_standard_loader import (
    load_gold_standard, get_binary_labels, DIMENSIONS
)


def compute_quality_metrics(
    results: List[ExperimentResult],
    gold_standard_path: str = "data/gold_standard/gold_standard.csv"
) -> Dict:
    """
    Compute precision, recall, F1 per strategy using run 0 only.
    Run 0 is the primary quality measurement run.
    Runs 1 and 2 are used for consistency measurement.

    Args:
        results:             list of all ExperimentResult objects
        gold_standard_path:  path to gold standard CSV

    Returns:
        dict keyed by strategy name with quality metrics
    """
    df = load_gold_standard(gold_standard_path)
    metrics = {}

    # Group results by strategy, use run 0 only
    strategies = set(r.strategy for r in results)

    for strategy in strategies:
        strategy_results = [
            r for r in results
            if r.strategy == strategy and r.run == 0
        ]

        all_gold = []
        all_pred = []

        for result in strategy_results:
            try:
                gold_labels = get_binary_labels(df, result.doc_id)
            except ValueError:
                continue

            for dim in DIMENSIONS:
                gold_val = gold_labels.get(dim)
                if gold_val is None:
                    # Skip N/A dimensions
                    continue

                pred_val = result.is_flagged(dim)
                all_gold.append(gold_val)
                all_pred.append(pred_val)

        if not all_gold:
            continue

        metrics[strategy] = {
            "precision": round(precision_score(
                all_gold, all_pred,
                average="macro", zero_division=0), 3),
            "recall": round(recall_score(
                all_gold, all_pred,
                average="macro", zero_division=0), 3),
            "f1": round(f1_score(
                all_gold, all_pred,
                average="macro", zero_division=0), 3),
            "total_judgements": len(all_gold),
            "correct":          sum(
                1 for g, p in zip(all_gold, all_pred) if g == p),
            "accuracy": round(
                sum(1 for g, p in zip(all_gold, all_pred) if g == p)
                / len(all_gold), 3),
        }

    return metrics