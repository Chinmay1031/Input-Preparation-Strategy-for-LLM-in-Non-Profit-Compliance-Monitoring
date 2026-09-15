"""
Counts tokens for each strategy and returns all prepared inputs.
"""

import tiktoken
from src.ingestion.document_schema import ParsedDocument
from src.strategies.s1_full_text import prepare_s1
from src.strategies.s2_section_filter import prepare_s2
from src.strategies.s3_field_extractor import prepare_s3
from src.strategies.s4_hybrid import prepare_s4
from src.strategies.s5_extended import prepare_s5

ENCODING = tiktoken.encoding_for_model("gpt-4o")


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    return len(ENCODING.encode(text))


def get_all_strategies(doc: ParsedDocument) -> dict:
    """
    Prepare all five strategy inputs for a document.
    Returns a dict with text and token count for each strategy.

    Strategies are ordered from least to most compressed:
      S1  full document text, no preparation
      S2  compliance-relevant sections only
      S5  extended verbatim extraction, targets ~50% reduction
      S4  structured extraction plus narrative context
      S3  structured verbatim extraction only
    """
    strategies = {
        "S1_full":     prepare_s1(doc),
        "S2_sections": prepare_s2(doc),
        "S3_fields":   prepare_s3(doc),
        "S4_hybrid":   prepare_s4(doc),
        "S5_extended": prepare_s5(doc),
    }
 
    results = {}
    for name, text in strategies.items():
        results[name] = {
            "text":        text,
            "token_count": count_tokens(text),
            "char_count":  len(text),
        }

    return results