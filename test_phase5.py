"""
test_phase5.py
--------------
Runs the full experiment across all documents in data/pdfs/
with all four strategies and three runs each.

Documents exceeding the per-request token limit are truncated by the
caller and the truncation is recorded in the results, so the practical
infeasibility of full-document prompting on large documents becomes
measurable data rather than a failure.

Saves all results to data/results/all_results.json.
"""

import sys, os, json, time
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

# Pause between API calls to stay within the tokens-per-minute budget
INTER_CALL_DELAY = 3

# ── Run full experiment ───────────────────────────────────────────────────────
pdf_files = list(Path("data/pdfs").glob("*.pdf"))
total_calls = len(pdf_files) * 4 * N_RUNS

print(f"Found {len(pdf_files)} document(s)")
print(f"Strategies: 4 | Runs per strategy: {N_RUNS}")
print(f"Total API calls: {total_calls}")
print(f"Estimated runtime: ~{(total_calls * (INTER_CALL_DELAY + 3)) // 60} minutes\n")

all_results    = []
truncated_docs = []

for pdf_path in pdf_files:
    doc = parse_document(str(pdf_path), pdf_path.stem)
    print(f"\nProcessing: {doc.doc_id}")
    print(f"Type: {doc.document_type} | Pages: {doc.total_pages}")

    strategies = get_all_strategies(doc)

    for strategy_name, strategy_data in strategies.items():
        prepared_text = strategy_data["text"]
        token_count   = strategy_data["token_count"]
        print(f"\n  Strategy: {strategy_name} ({token_count:,} tokens)")

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

            # Record and report truncation
            trunc_note = ""
            if raw_result.get("was_truncated"):
                original = raw_result.get("original_tokens", 0)
                trunc_note = f" [TRUNCATED from {original:,}]"
                truncated_docs.append({
                    "doc_id":          doc.doc_id,
                    "strategy":        strategy_name,
                    "run":             run,
                    "original_tokens": original,
                    "sent_tokens":     result.tokens_input,
                })

            print(f"✓ tokens={result.tokens_input:,}{trunc_note} "
                  f"faith={result.faithfulness_score:.0%}")

            time.sleep(INTER_CALL_DELAY)

# ── Save results ──────────────────────────────────────────────────────────────
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
serialised = [r.to_dict() for r in all_results]
with open(RESULTS_PATH, "w") as f:
    json.dump(serialised, f, indent=2)

print(f"\nResults saved to {RESULTS_PATH}")
print(f"Total results: {len(all_results)}")

# ── Report truncation ─────────────────────────────────────────────────────────
if truncated_docs:
    trunc_path = Path("data/results/truncation_log.json")
    with open(trunc_path, "w") as f:
        json.dump(truncated_docs, f, indent=2)

    print(f"\n{'='*70}")
    print("TRUNCATION REPORT")
    print(f"{'='*70}")
    print(f"{len(truncated_docs)} of {len(all_results)} calls required "
          f"truncation to fit within the API request limit.")

    affected = {}
    for t in truncated_docs:
        key = (t["doc_id"], t["strategy"])
        affected[key] = t["original_tokens"]

    for (doc_id, strategy), original in sorted(affected.items()):
        print(f"  {doc_id[:45]:<45} {strategy:<14} "
              f"{original:>8,} tokens")

    print(f"\nTruncation log saved to {trunc_path}")
    print("This limitation applies only to full-document strategies and "
          "is reported in the results chapter.")

# ── Generate master table ─────────────────────────────────────────────────────
print("\nGenerating master results table...")
try:
    df = generate_master_table(all_results, GOLD_PATH)
    print_master_table(df)
except Exception as e:
    print(f"Table generation error: {e}")
    print("Results are saved — run compute_metrics.py separately.")

print("\n✅ Phase 5 test complete.")