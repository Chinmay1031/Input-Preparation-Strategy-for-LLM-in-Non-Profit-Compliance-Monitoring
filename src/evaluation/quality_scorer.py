
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
    df = load_gold_standard(gold_standard_path)
    metrics = {}

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