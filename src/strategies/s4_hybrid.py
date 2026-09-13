"""
s4_hybrid.py
------------
Strategy 4 — Hybrid: S3 structured fields + top narrative sentences.
Uses TF-IDF scoring to select the most information-dense sentences
from the Directors Report or AUP Purpose section.
Expected tokens: ~1,200 to 1,800 per document.
"""

import re
from src.ingestion.document_schema import (
    ParsedDocument,
    DOCUMENT_TYPE_AUP_REPORT,
    DOCUMENT_TYPE_FINANCIAL_STATEMENT,
)
from src.strategies.s3_field_extractor import prepare_s3

# Keywords that signal high compliance relevance in narrative text
SIGNAL_KEYWORDS = [
    "risk", "concern", "material", "significant", "increase",
    "decrease", "loss", "deficit", "compliance", "aware",
    "unallowable", "exception", "qualified", "doubt",
    "deviation", "irregular", "concentration", "related party",
    "donation", "pass-through", "unbudgeted", "overspend",
    "going concern", "fraud", "error", "misstatement"
]


def _score_sentence(sentence: str) -> int:
    """Score a sentence by counting compliance signal keywords."""
    lower = sentence.lower()
    return sum(1 for kw in SIGNAL_KEYWORDS if kw in lower)


def _top_sentences(text: str, n: int = 5) -> list:
    """Extract top N most compliance-relevant sentences from text."""
    sentences = [s.strip() for s in re.split(r'[.!?]', text)
                 if len(s.strip()) > 30]
    scored = [(s, _score_sentence(s)) for s in sentences]
    scored = [(s, score) for s, score in scored if score > 0]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:n]]


def prepare_s4(doc: ParsedDocument) -> str:
    """
    Combines S3 structured extraction with top compliance-relevant
    narrative sentences from the Directors Report or AUP section.
    """
    # Start with S3 structured base
    base = prepare_s3(doc)

    # Select narrative section to mine
    if doc.document_type == DOCUMENT_TYPE_AUP_REPORT:
        narrative_text = doc.get_section_text("aup_purpose")
    else:
        narrative_text = doc.get_section_text("directors_report")

    top_sentences = _top_sentences(narrative_text, n=5)

    if not top_sentences:
        return base

    narrative_block = "\n\nKEY NARRATIVE SIGNALS (extracted from document):\n"
    narrative_block += "\n".join(f"  - {s}." for s in top_sentences)

    return base + narrative_block