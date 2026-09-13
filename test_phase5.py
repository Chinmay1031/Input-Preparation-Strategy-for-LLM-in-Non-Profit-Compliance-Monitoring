"""
test_phase5.py
--------------
Tests Phase 5 evaluation framework on saved results.
Runs the full pipeline on both documents with all 4 strategies
and 3 runs each — total 24 API calls.
Estimated cost: approximately $0.15 to $0.25.
Saves all results to data/results/all_results.json.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion import parse_document
from src.strategies import get_all_strategies
from src.llm import call_llm_with_retry, N_RUNS
from src.output_processing import (
    normalise_verdict, compute_faithfulness
)
from src.evaluation import (
    generate_master_table, print_master_table
)
from pathlib import Path

RESULTS_PATH = Path("data/results/all_results.json")
GOLD_PATH    = "data/gold_standard/gold_standard.csv"

# ── Run full experiment ───────────────────────────────────────────────────────
pdf_files = list(Path("data/pdfs").glob("*.pdf"))
print(f"Found {len(pdf_files)} document(s)")
print(f"Strategies: 4 | Runs per strategy: {N_RUNS}")
print(f"Total API calls: {len(pdf_files) * 4 * N_RUNS}\n")

all_results = []

for pdf_path in pdf_files:
    doc = parse_document(str(pdf_path), pdf_path.stem)
    print(f"\nProcessing: {doc.doc_id}")
    print(f"Type: {doc.document_type} | Pages: {doc.total_pages}")

    strategies = get_all_strategies(doc)

    for strategy_name, strategy_data in strategies.items():
        prepared_text = strategy_data["text"]
        token_count   = strategy_data["token_count"]
        print(f"\n  Strategy: {strategy_name} ({token_count} tokens)")

        for run in range(N_RUNS):
            print(f"    Run {run + 1}/{N_RUNS}...", end=" ", flush=True)

            raw_result = call_llm_with_retry(prepared_text)

            result = normalise_verdict(
                raw_result,
                doc_id=doc.doc_id,
                strategy=strategy_name,
                run=run
            )

            result = compute_faithfulness(
                result,
                source_text=doc.full_text,
                prepared_text=prepared_text
            )

            all_results.append(result)
            print(f"✓ tokens={result.tokens_input} "
                  f"faith={result.faithfulness_score:.0%}")

# ── Save results ──────────────────────────────────────────────────────────────
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
serialised = [r.to_dict() for r in all_results]
with open(RESULTS_PATH, "w") as f:
    json.dump(serialised, f, indent=2)
print(f"\nResults saved to {RESULTS_PATH}")
print(f"Total results: {len(all_results)}")

# ── Generate master table ─────────────────────────────────────────────────────
print("\nGenerating master results table...")
try:
    df = generate_master_table(all_results, GOLD_PATH)
    print_master_table(df)
except Exception as e:
    print(f"Table generation error: {e}")
    print("Results are saved — run compute_metrics.py separately.")

print("\n✅ Phase 5 test complete.")