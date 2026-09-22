
from src.ingestion.document_schema import (
    ParsedDocument,
    DOCUMENT_TYPE_AUP_REPORT,
    DOCUMENT_TYPE_FINANCIAL_STATEMENT,
    SEC_AUDIT_REPORT, SEC_DIRECTORS_REPORT,
    SEC_FINANCIAL_STATEMENTS, SEC_NOTES, SEC_DETAILED_INCOME,
    SEC_AUP_PURPOSE, SEC_AUP_PROCEDURES, SEC_AUP_ANNEXURE_A,
    SEC_AUP_ANNEXURE_B, SEC_AUP_NOTES
)

RELEVANT_FINANCIAL = [
    SEC_DIRECTORS_REPORT,
    SEC_FINANCIAL_STATEMENTS,
    SEC_DETAILED_INCOME,
    SEC_NOTES,
    SEC_AUDIT_REPORT,
]

RELEVANT_AUP = [
    SEC_AUP_PURPOSE,
    SEC_AUP_PROCEDURES,
    SEC_AUP_ANNEXURE_A,
    SEC_AUP_NOTES,
]


def prepare_s2(doc: ParsedDocument) -> str:
    if doc.document_type == DOCUMENT_TYPE_AUP_REPORT:
        relevant_sections = RELEVANT_AUP
    else:
        relevant_sections = RELEVANT_FINANCIAL

    parts = []
    for sec_type in relevant_sections:
        if doc.has_section(sec_type):
            text = doc.get_section_text(sec_type).strip()
            if text:
                parts.append(
                    f"=== {sec_type.upper().replace('_', ' ')} ===\n{text}"
                )

    return "\n\n".join(parts).strip()