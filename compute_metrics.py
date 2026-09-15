"""
Reads saved results from data/results/all_results.json and computes all
four evaluation metrics, without making API calls. Run any time to
regenerate the thesis results table.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from src.output_processing.result_schema import (
    ExperimentResult, ComplianceFlag
)
from src.evaluation import (
    generate_master_table, print_master_table,
    compute_quality_metrics, compute_efficiency_metrics,
    compute_consistency_metrics, compute_faithfulness_metrics
)

RESULTS_PATH = Path("data/results/all_results.json")
GOLD_PATH    = "data/gold_standard/gold_standard.csv"


def load_results(path: Path) -> list:
    """Rebuild ExperimentResult objects from saved JSON."""
    with open(path) as f:
        data = json.load(f)

    results = []
    for d in data:
        flags = [
            ComplianceFlag(
                dimension=f.get("dimension", ""),
                severity=f.get("severity", "FLAG"),
                evidence=f.get("evidence", ""),
                is_faithful=f.get("is_faithful"),
                match_score=f.get("match_score"),
            )
            for f in d.get("flags", [])
        ]

        results.append(ExperimentResult(
            doc_id=d["doc_id"],
            strategy=d["strategy"],
            run=d["run"],
            revenue_concentration=d.get("revenue_concentration", "CLEAR"),
            expense_spike=d.get("expense_spike", "CLEAR"),
            passthrough_risk=d.get("passthrough_risk", "CLEAR"),
            unallowable_expenditure=d.get("unallowable_expenditure", "CLEAR"),
            audit_opinion=d.get("audit_opinion", "CLEAR"),
            going_concern=d.get("going_concern", "CLEAR"),
            overall_verdict=d.get("overall_verdict", "COMPLIANT"),
            confidence=d.get("confidence", 0.0),
            flags=flags,
            tokens_input=d.get("tokens_input", 0),
            tokens_output=d.get("tokens_output", 0),
            tokens_total=d.get("tokens_total", 0),
            faithfulness_score=d.get("faithfulness_score", 1.0),
            hallucination_rate=d.get("hallucination_rate", 0.0),
            hallucinated_flags=d.get("hallucinated_flags", []),
            parse_error=d.get("parse_error", False),
            model=d.get("model", ""),
        ))

    return results


if not RESULTS_PATH.exists():
    print(f"No results found at {RESULTS_PATH}")
    print("Run test_phase5.py first to generate results.")
    sys.exit(1)

results = load_results(RESULTS_PATH)
print(f"Loaded {len(results)} results from {RESULTS_PATH}\n")

print("="*70)
print("QUALITY METRICS (vs gold standard)")
print("="*70)
quality = compute_quality_metrics(results, GOLD_PATH)
for strategy, m in sorted(quality.items()):
    print(f"\n  {strategy}")
    print(f"    Precision:        {m['precision']}")
    print(f"    Recall:           {m['recall']}")
    print(f"    F1:               {m['f1']}")
    print(f"    Accuracy:         {m['accuracy']}")
    print(f"    Judgements:       {m['total_judgements']}")
    print(f"    Correct:          {m['correct']}")

print("\n" + "="*70)
print("EFFICIENCY METRICS")
print("="*70)
efficiency = compute_efficiency_metrics(results)
for strategy, m in sorted(efficiency.items()):
    print(f"\n  {strategy}")
    print(f"    Avg input tokens: {m['avg_input_tokens']:,}")
    print(f"    Cost per doc:     ${m['cost_per_doc_usd']}")
    print(f"    Cost per 100:     ${m['cost_per_100_usd']}")
    print(f"    Reduction vs S1:  {m['token_reduction_vs_s1']}")

print("\n" + "="*70)
print("CONSISTENCY METRICS (Cohen's Kappa across 3 runs)")
print("="*70)
consistency = compute_consistency_metrics(results)
for strategy, m in sorted(consistency.items()):
    print(f"\n  {strategy}")
    print(f"    Cohen's Kappa:    {m['kappa']}")
    print(f"    Interpretation:   {m['kappa_interpretation']}")
    print(f"    Agreement rate:   {m['agreement_rate']}")

print("\n" + "="*70)
print("FAITHFULNESS METRICS")
print("="*70)
faithfulness = compute_faithfulness_metrics(results)
for strategy, m in sorted(faithfulness.items()):
    print(f"\n  {strategy}")
    print(f"    Avg faithfulness: {m['avg_faithfulness']}")
    print(f"    Hallucination:    {m['hallucination_pct']}%")
    print(f"    Total flags:      {m['total_flags_raised']}")
    print(f"    Hallucinated:     {m['total_hallucinated']}")

df = generate_master_table(results, GOLD_PATH)
print_master_table(df)

out_path = Path("data/results/master_results_table.csv")
df.to_csv(out_path, index=False)
print(f"\nMaster table saved to {out_path}")