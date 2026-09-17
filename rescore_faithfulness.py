
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from src.ingestion import parse_document
from src.strategies import get_all_strategies
from src.output_processing.result_schema import (
    ExperimentResult, ComplianceFlag
)
from src.output_processing import compute_faithfulness

RESULTS_PATH = Path("data/results/all_results.json")

if not RESULTS_PATH.exists():
    print(f"No results found at {RESULTS_PATH}")
    sys.exit(1)

with open(RESULTS_PATH) as f:
    data = json.load(f)

print(f"Loaded {len(data)} results\n")

                                                                  
doc_cache = {}
for pdf_path in Path("data/pdfs").glob("*.pdf"):
    doc_id = pdf_path.stem
    print(f"Parsing {doc_id}...")
    doc = parse_document(str(pdf_path), doc_id)
    doc_cache[doc_id] = {
        "full_text":  doc.full_text,
        "strategies": get_all_strategies(doc),
    }

print(f"\nRescoring {len(data)} results...\n")

rescored = []
for d in data:
    doc_id   = d["doc_id"]
    strategy = d["strategy"]

    if doc_id not in doc_cache:
        print(f"  Skipping {doc_id} — document not found")
        rescored.append(d)
        continue

    flags = [
        ComplianceFlag(
            dimension=f.get("dimension", ""),
            severity=f.get("severity", "FLAG"),
            evidence=f.get("evidence", ""),
        )
        for f in d.get("flags", [])
    ]

    result = ExperimentResult(
        doc_id=doc_id,
        strategy=strategy,
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
        parse_error=d.get("parse_error", False),
        model=d.get("model", ""),
    )

    cache         = doc_cache[doc_id]
    prepared_text = cache["strategies"].get(strategy, {}).get("text", "")

    result = compute_faithfulness(
        result,
        source_text=cache["full_text"],
        prepared_text=prepared_text,
    )

    rescored.append(result.to_dict())

with open(RESULTS_PATH, "w") as f:
    json.dump(rescored, f, indent=2)

print(f"Rescored results saved to {RESULTS_PATH}\n")

print("="*72)
print("CITATION BEHAVIOUR BY STRATEGY")
print("="*72)

by_strategy = {}
for d in rescored:
    if d["run"] != 0:
        continue
    s = d["strategy"]
    by_strategy.setdefault(s, {
        "verbatim": [], "synthesis": [], "hallucination": [],
        "mean_score": [], "total_flags": 0, "synth_examples": [],
    })
    b = by_strategy[s]
    b["verbatim"].append(d.get("verbatim_rate", 1.0))
    b["synthesis"].append(d.get("synthesis_rate", 0.0))
    b["hallucination"].append(d.get("hallucination_rate", 0.0))
    b["mean_score"].append(d.get("mean_match_score", 1.0))
    b["total_flags"] += len(d.get("flags", []))
    b["synth_examples"].extend(d.get("synthesised_flags", []))

order = ["S1_full", "S2_sections", "S5_extended", "S4_hybrid", "S3_fields"]
for s in order:
    if s not in by_strategy:
        continue
    b = by_strategy[s]
    n = len(b["verbatim"])
    print(f"\n  {s}")
    print(f"    Total flags raised:     {b['total_flags']}")
    print(f"    Verbatim citation:      "
          f"{sum(b['verbatim'])/n:.1%}")
    print(f"    Synthesised claim:      "
          f"{sum(b['synthesis'])/n:.1%}")
    print(f"    Unsupported:            "
          f"{sum(b['hallucination'])/n:.1%}")
    print(f"    Mean match score:       "
          f"{sum(b['mean_score'])/n:.3f}")

    if b["synth_examples"]:
        print(f"    Synthesised examples:")
        for ex in b["synth_examples"][:2]:
            print(f"      {ex[:90]}")

print("\n" + "="*72)
print("\nRun compute_metrics.py to regenerate the master table.")