
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion import parse_document
from src.strategies import get_all_strategies
from src.llm import call_llm_with_retry
from pathlib import Path

pdf_path = list(Path("data/pdfs").glob("*.pdf"))[0]
print(f"Testing on: {pdf_path.name}\n")

doc        = parse_document(str(pdf_path), pdf_path.stem)
strategies = get_all_strategies(doc)

strategy_name = "S3_fields"
prepared_text = strategies[strategy_name]["text"]
token_count   = strategies[strategy_name]["token_count"]

print(f"Strategy:     {strategy_name}")
print(f"Input tokens: {token_count}")
print(f"Calling GPT-4o...\n")

result = call_llm_with_retry(prepared_text)

print(f"Tokens used (API): {result['tokens_input']} input / "
      f"{result['tokens_output']} output")
print(f"Model: {result['model']}\n")

print("── Compliance Verdict ──────────────────────────────────")
verdict = result["verdict"]
dimensions = [
    "revenue_concentration", "expense_spike", "passthrough_risk",
    "unallowable_expenditure", "audit_opinion", "going_concern"
]
for dim in dimensions:
    label = verdict.get(dim, "UNKNOWN")
    print(f"  {dim:<30} {label}")

print(f"\n  Overall verdict: {verdict.get('overall_verdict', 'UNKNOWN')}")
print(f"  Confidence:      {verdict.get('confidence', 0)}")

if verdict.get("flags"):
    print(f"\n── Flags raised ────────────────────────────────────────")
    for flag in verdict["flags"]:
        print(f"  [{flag.get('severity')}] {flag.get('dimension')}")
        print(f"    Evidence: {flag.get('evidence')}")
else:
    print("\n  No flags raised.")

print("\n── Raw JSON (for debugging) ─────────────────────────────")
print(json.dumps(verdict, indent=2))

print("\nPhase 3 test complete.")