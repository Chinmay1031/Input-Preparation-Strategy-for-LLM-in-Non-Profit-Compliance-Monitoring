"""
token_counter.py
----------------
Counts tokens for each strategy and returns all four prepared inputs.
Uses tiktoken — the same tokeniser OpenAI uses internally.
"""

import tiktoken
from src.ingestion.document_schema import ParsedDocument
from src.strategies.s1_full_text import prepare_s1
from src.strategies.s2_section_filter import prepare_s2
from src.strategies.s3_field_extractor import prepare_s3
from src.strategies.s4_hybrid import prepare_s4

# Use GPT-4o encoding
ENCODING = tiktoken.encoding_for_model("gpt-4o")


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    return len(ENCODING.encode(text))


def get_all_strategies(doc: ParsedDocument) -> dict:
    """
    Prepare all four strategy inputs for a document.
    Returns a dict with text and token count for each strategy.
    """
    strategies = {
        "S1_full":     prepare_s1(doc),
        "S2_sections": prepare_s2(doc),
        "S3_fields":   prepare_s3(doc),
        "S4_hybrid":   prepare_s4(doc),
    }

    results = {}
    for name, text in strategies.items():
        results[name] = {
            "text":        text,
            "token_count": count_tokens(text),
            "char_count":  len(text),
        }

    return results