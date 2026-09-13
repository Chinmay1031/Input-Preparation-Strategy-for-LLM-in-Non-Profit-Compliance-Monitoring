"""
test_phase4.py
--------------
Tests Phase 4 output processing on one document with one strategy.
Runs the full pipeline: parse -> strategy -> LLM -> normalise -> faithfulness.
Costs approximately $0.01.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion import parse_document
from src.strategies import get_all_strategies
from src.llm import call_llm_with_retry
from src.output_processing import (
    normalise_verdict,
    compute_faithfulness,
    detect_hallucinations,
    print_hallucination_report
)
from pathlib import Path

# ── Run on first PDF with S3 ──────────────────────────────────────────────────
pdf_path = list(Path("data/pdfs").glob("*.pdf"))[0]
print(f"Testing on: {pdf_path.name}\n")

doc        = parse_document(str(pdf_path), pdf_path.stem)
strategies = get_all_strategies(doc)

strategy_name = "S3_fields"
prepared_text = strategies[strategy_name]["text"]

print(f"Calling LLM with {strategy_name} "
      f"({strategies[strategy_name]['token_count']} tokens)...\n")

# Phase 3 — LLM call
raw_result = call_llm_with_retry(prepared_text)

# Phase 4a — normalise verdict
result = normalise_verdict(
    raw_result,
    doc_id=doc.doc_id,
    strategy=strategy_name,
    run=0
)

print("── Normalised Verdict ───────────────────────────────────")
for dim, label in result.get_dimension_labels().items():
    print(f"  {dim:<30} {label}")
print(f"  {'overall_verdict':<30} {result.overall_verdict}")
print(f"  {'confidence':<30} {result.confidence}")

# Phase 4b — faithfulness check
result = compute_faithfulness(result, doc.full_text, prepared_text)

print(f"\n── Faithfulness Results ─────────────────────────────────")
print(f"  Faithfulness score:   {result.faithfulness_score:.2%}")
print(f"  Hallucination rate:   {result.hallucination_rate:.2%}")
print(f"  Total flags:          {len(result.flags)}")

if result.flags:
    print(f"\n  Flag-by-flag breakdown:")
    for flag in result.flags:
        status = "GROUNDED" if flag.is_faithful else "HALLUCINATED"
        print(f"    [{status}] {flag.dimension}")
        print(f"      Evidence:    {flag.evidence[:80]}")
        print(f"      Match score: {flag.match_score:.2%}")

if result.hallucinated_flags:
    print(f"\n  Hallucinated evidence:")
    for h in result.hallucinated_flags:
        print(f"    {h}")

# Phase 4c — hallucination detector summary
summary = detect_hallucinations([result])
print_hallucination_report(summary)

# Show serialised result
print(f"\n── Serialised Result (sample) ───────────────────────────")
result_dict = result.to_dict()
print(json.dumps({
    "doc_id":             result_dict["doc_id"],
    "strategy":           result_dict["strategy"],
    "faithfulness_score": result_dict["faithfulness_score"],
    "hallucination_rate": result_dict["hallucination_rate"],
    "tokens_input":       result_dict["tokens_input"],
}, indent=2))

print("\n✅ Phase 4 test complete.")