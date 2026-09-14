"""
s4_hybrid.py
------------
Strategy 4 — Hybrid: S3 verbatim extract plus narrative context.

Adds the most compliance-relevant sentences from the management or
directors report on top of the S3 structured extract. Tests whether
additional narrative context improves compliance reasoning beyond
the structured lines alone.

Expected tokens: ~900 to 1,500 per document.
"""

import re
from src.ingestion.document_schema import (
    ParsedDocument,
    DOCUMENT_TYPE_AUP_REPORT,
)
from src.strategies.s3_field_extractor import prepare_s3

SIGNAL_KEYWORDS = [
    "risk", "concern", "material", "significant", "increase",
    "decrease", "loss", "deficit", "compliance", "aware",
    "unallowable", "exception", "qualified", "doubt",
    "deviation", "irregular", "concentration", "related party",
    "donation", "pass-through", "unbudgeted", "overspend",
    "going concern", "fraud", "error", "misstatement",
    "dependent", "terminated", "funding", "restricted",
]

EXCLUDE_BOILERPLATE = [
    "our objectives are to obtain",
    "in performing an audit in accordance",
    "auditor's responsibilities for the audit",
    "the accompanying notes are an integral",
]


def _score_sentence(sentence: str) -> int:
    low = sentence.lower()
    return sum(1 for kw in SIGNAL_KEYWORDS if kw in low)


def _top_sentences(text: str, n: int = 6) -> list:
    """Extract the most compliance-relevant sentences from text."""
    if not text:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
    scored = []

    for s in sentences:
        s = re.sub(r'\s{2,}', ' ', s.strip())
        low = s.lower()

        if len(s) < 50 or len(s) > 350:
            continue
        if any(x in low for x in EXCLUDE_BOILERPLATE):
            continue

        score = _score_sentence(s)
        if score > 0:
            scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:n]]


def prepare_s4(doc: ParsedDocument) -> str:
    """
    S4 — S3 verbatim extract plus additional narrative context
    drawn from the management or directors report.
    """
    base = prepare_s3(doc)

    if doc.document_type == DOCUMENT_TYPE_AUP_REPORT:
        narrative = doc.get_section_text("aup_purpose")
    else:
        narrative = (doc.get_section_text("directors_report")
                     or doc.get_section_text("notes")[:8000])

    top_sentences = _top_sentences(narrative, n=6)

    if not top_sentences:
        return base

    block = ["", "[ADDITIONAL NARRATIVE CONTEXT — VERBATIM]"]
    for s in top_sentences:
        block.append(f"  {s}")

    return base + "\n" + "\n".join(block)