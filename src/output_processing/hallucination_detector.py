"""
hallucination_detector.py
-------------------------
Logs and summarises hallucinations detected across all experiment runs.
Used to generate Figure 3 in the thesis results chapter.
"""

from typing import List, Dict
from .result_schema import ExperimentResult


def detect_hallucinations(results: List[ExperimentResult]) -> Dict:
    """
    Summarise hallucination patterns across a list of results.

    Args:
        results: list of ExperimentResult objects with faithfulness
                 scores already computed

    Returns:
        dict with hallucination summary statistics per strategy
    """
    summary = {}

    for result in results:
        strategy = result.strategy

        if strategy not in summary:
            summary[strategy] = {
                "total_flags":        0,
                "hallucinated_flags": 0,
                "faithfulness_scores": [],
                "hallucination_examples": [],
            }

        s = summary[strategy]
        s["total_flags"]          += len(result.flags)
        s["hallucinated_flags"]   += len(result.hallucinated_flags)
        s["faithfulness_scores"].append(result.faithfulness_score)

        for example in result.hallucinated_flags:
            s["hallucination_examples"].append({
                "doc_id":   result.doc_id,
                "run":      result.run,
                "evidence": example,
            })

    # Compute averages
    for strategy, s in summary.items():
        scores = s["faithfulness_scores"]
        s["avg_faithfulness"]    = sum(scores) / len(scores) if scores else 1.0
        s["avg_hallucination"]   = 1.0 - s["avg_faithfulness"]
        s["hallucination_rate_pct"] = round(s["avg_hallucination"] * 100, 1)

    return summary


def print_hallucination_report(summary: Dict) -> None:
    """Print a formatted hallucination report to the terminal."""
    print("\n── Hallucination Report ─────────────────────────────────")
    for strategy, s in summary.items():
        print(f"\n  Strategy: {strategy}")
        print(f"    Total flags raised:    {s['total_flags']}")
        print(f"    Hallucinated flags:    {s['hallucinated_flags']}")
        print(f"    Avg faithfulness:      {s['avg_faithfulness']:.2%}")
        print(f"    Hallucination rate:    {s['hallucination_rate_pct']}%")

        if s["hallucination_examples"]:
            print(f"    Examples:")
            for ex in s["hallucination_examples"][:3]:
                print(f"      [{ex['doc_id']} run {ex['run']}]")
                print(f"      {ex['evidence'][:80]}")