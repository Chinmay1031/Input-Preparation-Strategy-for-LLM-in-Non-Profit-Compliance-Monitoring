
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion import parse_document
from src.strategies import get_all_strategies
from pathlib import Path

pdf_dir   = Path("data/pdfs")
pdf_files = list(pdf_dir.glob("*.pdf"))

for pdf_path in pdf_files:
    doc = parse_document(str(pdf_path), pdf_path.stem)
    print(f"\n{'='*60}")
    print(f"Document: {doc.doc_id}")
    print(f"Type:     {doc.document_type}")
    print(f"{'='*60}")

    strategies = get_all_strategies(doc)

    print(f"\n{'Strategy':<15} {'Tokens':>8} {'Chars':>8}")
    print("-" * 35)
    for name, data in strategies.items():
        print(f"{name:<15} {data['token_count']:>8,} {data['char_count']:>8,}")

    print(f"\n── S3 Field Extraction Preview ──")
    print(strategies["S3_fields"]["text"][:800])
    print("...")

print("\nPhase 2 test complete.")