"""
test_phase1.py
--------------
Auto-detects whatever PDFs are in data/pdfs/ and parses them.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion import parse_document, summarise_budget_lines
from pathlib import Path

# Auto-detect all PDFs in the folder
pdf_dir = Path("data/pdfs")
pdf_files = list(pdf_dir.glob("*.pdf"))

if not pdf_files:
    print("⚠️  No PDFs found in data/pdfs/")
    print("    Copy your PDF files there and run again.")
    sys.exit(1)

print(f"Found {len(pdf_files)} PDF(s) in data/pdfs/\n")

for pdf_path in pdf_files:
    doc_id = pdf_path.stem
    print(f"\n{'='*60}")
    print(f"PARSING: {doc_id}")
    print('='*60)

    doc = parse_document(str(pdf_path), doc_id)
    print(doc.summary())

    print(f"\n── Sections found ({'─'*30})")
    for sec_type, section in doc.sections.items():
        print(
            f"  {sec_type:<35} "
            f"pages {section.page_start}–{section.page_end}  "
            f"({len(section.raw_text):,} chars, "
            f"{len(section.tables)} tables)"
        )

    print(f"\n── Financial figures ({'─'*28})")
    for key, val in doc.financial_figures.items():
        print(f"  {key:<35} {val}")

    if doc.budget_lines:
        print(f"\n── Budget lines extracted: {len(doc.budget_lines)}")
        summary = summarise_budget_lines(doc.budget_lines)
        print(f"  Total actual:      €{summary.get('total_actual', 0):>12,.2f}")
        print(f"  Total budget:      €{summary.get('total_budget', 0):>12,.2f}")
        print(f"  Overspend items:   {summary.get('overspend_count', 0)}")
        print(f"  Unbudgeted items:  {summary.get('unbudgeted_count', 0)}")

        if summary.get('top_overspends'):
            print(f"\n  ── Top overspends:")
            for bl in summary['top_overspends']:
                pct = f"{bl.variance_pct:.0%}" if bl.variance_pct else "N/A"
                print(
                    f"    {bl.category[:45]:<45} "
                    f"actual={bl.actual:>10,.0f}  "
                    f"budget={bl.budget:>10,.0f}  "
                    f"variance={pct}"
                )

    if doc.parse_warnings:
        print(f"\n⚠️  Warnings:")
        for w in doc.parse_warnings:
            print(f"   {w}")

    print(f"\n── OCR quality: {doc.ocr_quality_score:.2f}")
    print(f"── Full text length: {len(doc.full_text):,} characters")

print("\n✅ Phase 1 test complete.")