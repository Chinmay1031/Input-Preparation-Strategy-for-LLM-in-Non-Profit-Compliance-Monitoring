"""
Strategy 1 — Naive baseline. Passes the complete OCR-extracted text to
the LLM unmodified. ~4,000 to 9,000 tokens per document.
"""

from src.ingestion.document_schema import ParsedDocument


def prepare_s1(doc: ParsedDocument) -> str:
    """
    Returns the full extracted text of the document.
    No filtering, no compression, no domain knowledge applied.
    """
    return doc.full_text.strip()