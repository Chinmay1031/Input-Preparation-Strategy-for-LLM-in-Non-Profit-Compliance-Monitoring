"""
Aggregates faithfulness scores across all results per strategy.
Used to generate Figure 3 in the thesis results chapter.
"""

from typing import List, Dict
from src.output_processing.result_schema import ExperimentResult


def compute_faithfulness_metrics(
    results: List[ExperimentResult]
) -> Dict:
    """
    Aggregate faithfulness scores per strategy using run 0.

    Returns:
        dict keyed by strategy with faithfulness metrics
    """
    metrics = {}
    strategies = set(r.strategy for r in results)

    for strategy in strategies:
        strategy_results = [
            r for r in results
            if r.strategy == strategy and r.run == 0
        ]

        if not strategy_results:
            continue

        scores = [r.faithfulness_score for r in strategy_results]
        rates  = [r.hallucination_rate for r in strategy_results]
        total_flags = sum(len(r.flags) for r in strategy_results)
        hallucinated = sum(
            len(r.hallucinated_flags) for r in strategy_results
        )

        metrics[strategy] = {
            "avg_faithfulness":       round(
                sum(scores) / len(scores), 3),
            "avg_hallucination_rate": round(
                sum(rates) / len(rates), 3),
            "hallucination_pct":      round(
                sum(rates) / len(rates) * 100, 1),
            "total_flags_raised":     total_flags,
            "total_hallucinated":     hallucinated,
        }

    return metrics