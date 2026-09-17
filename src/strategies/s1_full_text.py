
from src.ingestion.document_schema import ParsedDocument


def prepare_s1(doc: ParsedDocument) -> str:
    return doc.full_text.strip()