"""
efficiency_scorer.py
--------------------
Computes token efficiency metrics for each strategy.
Token counts come directly from the OpenAI API response.

This is the most objective measurement in the thesis —
completely deterministic, no subjectivity involved.
"""

from typing import List, Dict
from src.output_processing.result_schema import ExperimentResult

# GPT-4o pricing as of mid-2025 — update if needed
COST_PER_1K_INPUT_TOKENS  = 0.005   # USD
COST_PER_1K_OUTPUT_TOKENS = 0.015   # USD


def compute_efficiency_metrics(
    results: List[ExperimentResult]
) -> Dict:
    """
    Compute token efficiency metrics per strategy.
    Uses run 0 only for token counting.

    Returns:
        dict keyed by strategy with efficiency metrics
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

        input_tokens  = [r.tokens_input  for r in strategy_results]
        output_tokens = [r.tokens_output for r in strategy_results]

        avg_input  = sum(input_tokens)  / len(input_tokens)
        avg_output = sum(output_tokens) / len(output_tokens)

        cost_per_doc = (
            (avg_input  / 1000) * COST_PER_1K_INPUT_TOKENS +
            (avg_output / 1000) * COST_PER_1K_OUTPUT_TOKENS
        )

        metrics[strategy] = {
            "avg_input_tokens":   round(avg_input),
            "avg_output_tokens":  round(avg_output),
            "avg_total_tokens":   round(avg_input + avg_output),
            "cost_per_doc_usd":   round(cost_per_doc, 5),
            "cost_per_100_usd":   round(cost_per_doc * 100, 3),
        }

    # Compute reduction vs S1 baseline
    s1_tokens = metrics.get("S1_full", {}).get("avg_input_tokens", 0)
    for strategy in metrics:
        if s1_tokens > 0:
            strategy_tokens = metrics[strategy]["avg_input_tokens"]
            reduction = (s1_tokens - strategy_tokens) / s1_tokens
            metrics[strategy]["token_reduction_vs_s1"] = (
                f"{reduction:.0%}"
            )
        else:
            metrics[strategy]["token_reduction_vs_s1"] = "N/A"

    return metrics