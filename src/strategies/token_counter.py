
import tiktoken
from src.ingestion.document_schema import ParsedDocument
from src.strategies.s1_full_text import prepare_s1
from src.strategies.s2_section_filter import prepare_s2
from src.strategies.s3_field_extractor import prepare_s3
from src.strategies.s4_hybrid import prepare_s4
from src.strategies.s5_extended import prepare_s5

ENCODING = tiktoken.encoding_for_model("gpt-4o")


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def get_all_strategies(doc: ParsedDocument) -> dict:
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