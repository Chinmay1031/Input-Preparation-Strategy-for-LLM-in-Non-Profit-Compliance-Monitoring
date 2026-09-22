
import re
from src.ingestion.document_schema import (
    ParsedDocument,
    DOCUMENT_TYPE_AUP_REPORT,
    DOCUMENT_TYPE_FINANCIAL_STATEMENT,
)
from src.ingestion.budget_extractor import summarise_budget_lines

from src.strategies.s3_field_extractor import (
    _extract_lines,
    _extract_sentences,
    _clean,
    REVENUE_KEYWORDS,
    EXPENSE_KEYWORDS,
    PASSTHROUGH_KEYWORDS,
    CONCENTRATION_KEYWORDS,
    UNALLOWABLE_KEYWORDS,
    GOING_CONCERN_DOUBT,
    GOING_CONCERN_CLEAN,
    GOING_CONCERN_BOILERPLATE,
    SUBSEQUENT_EVENT_KEYWORDS,
    AUDIT_OPINION_KEYWORDS,
    RELATED_PARTY_KEYWORDS,
)

EXT_LINES_PER_DIMENSION     = 30
EXT_SENTENCES_PER_DIMENSION = 12

FULL_SECTIONS_FINANCIAL = [
    "audit_report",
    "financial_statements",
    "detailed_income",
]

FULL_SECTIONS_AUP = [
    "aup_purpose",
    "aup_notes",
]

MAX_SECTION_CHARS = 9_000


def _section_block(doc: ParsedDocument, section_type: str) -> list:
    text = doc.get_section_text(section_type).strip()
    if not text:
        return []

    truncated = text[:MAX_SECTION_CHARS]
    was_cut = len(text) > MAX_SECTION_CHARS

    label = section_type.upper().replace("_", " ")
    block = [f"[{label} — FULL SECTION]"]

    for raw in truncated.split("\n"):
        line = _clean(raw)
        if len(line) >= 5:
            block.append(f"  {line}")

    if was_cut:
        block.append("  [section truncated]")

    block.append("")
    return block


def prepare_s5(doc: ParsedDocument) -> str:
    out  = []
    text = doc.full_text

    out.append("GRANTEE COMPLIANCE EXTRACT — EXTENDED")
    out.append(f"Grantee:       {doc.grantee_name or 'Not identified'}")
    out.append(f"Document type: {doc.document_type or 'Unknown'}")
    out.append(f"Fiscal year:   {doc.fiscal_year or 'Not identified'}")
    out.append(f"Auditor:       {doc.auditor or 'Not identified'}")
    out.append("")
    out.append("The following content is extracted verbatim from the source "
               "document. Assess each compliance dimension using only "
               "this evidence.")
    out.append("")

    if doc.document_type == DOCUMENT_TYPE_AUP_REPORT:
        out.append("[ENGAGEMENT TYPE]")
        out.append("  Agreed-Upon Procedures under ISRS 4400. "
                   "No audit opinion or assurance conclusion is expressed.")
        out.append("")

        figs = doc.financial_figures
        if "procedure_count" in figs or "no_exceptions_count" in figs:
            out.append("[PROCEDURE FINDINGS]")
            if "procedure_count" in figs:
                out.append(f"  Procedures performed: "
                           f"{int(figs['procedure_count'])}")
            if "no_exceptions_count" in figs:
                out.append(f"  Statements of 'no exceptions noted': "
                           f"{int(figs['no_exceptions_count'])}")
            out.append("")

        if doc.budget_lines:
            summary = summarise_budget_lines(doc.budget_lines)
            out.append("[BUDGET VS ACTUAL]")
            if summary.get("overspend_count"):
                out.append(f"  Line items over budget by >10%: "
                           f"{summary['overspend_count']}")
            if summary.get("unbudgeted_count"):
                out.append(f"  Unbudgeted line items: "
                           f"{summary['unbudgeted_count']}")
            out.append("")

        for section in FULL_SECTIONS_AUP:
            out.extend(_section_block(doc, section))

        findings = _extract_lines(
            doc.full_text,
            ["exception", "difference", "variance", "overspend",
             "falls out the scope", "reallocat"],
            require_figure=False,
            max_lines=EXT_LINES_PER_DIMENSION,
        )
        if findings:
            out.append("[AUDITOR FINDINGS — VERBATIM]")
            for f in findings:
                out.append(f"  {f}")
            out.append("")

    else:
        for section in FULL_SECTIONS_FINANCIAL:
            out.extend(_section_block(doc, section))

        revenue = _extract_lines(
            text, REVENUE_KEYWORDS, max_lines=EXT_LINES_PER_DIMENSION
        )
        if revenue:
            out.append("[REVENUE AND SUPPORT — VERBATIM]")
            for line in revenue:
                out.append(f"  {line}")
            out.append("")

        expenses = _extract_lines(
            text, EXPENSE_KEYWORDS, max_lines=EXT_LINES_PER_DIMENSION
        )
        if expenses:
            out.append("[EXPENSES — VERBATIM]")
            for line in expenses:
                out.append(f"  {line}")
            out.append("")

        passthrough = _extract_lines(
            text, PASSTHROUGH_KEYWORDS,
            require_figure=False, max_lines=EXT_LINES_PER_DIMENSION
        )
        if passthrough:
            out.append("[GRANTS PAID TO THIRD PARTIES — VERBATIM]")
            for line in passthrough:
                out.append(f"  {line}")
            out.append("")
        else:
            out.append("[GRANTS PAID TO THIRD PARTIES]")
            out.append("  No sub-grant or pass-through line items found "
                       "in this document.")
            out.append("")

        concentration = _extract_sentences(
            text, CONCENTRATION_KEYWORDS,
            max_sentences=EXT_SENTENCES_PER_DIMENSION
        )
        if concentration:
            out.append("[DONOR CONCENTRATION — VERBATIM]")
            for s in concentration:
                out.append(f"  {s}")
            out.append("")
        else:
            out.append("[DONOR CONCENTRATION]")
            out.append("  No explicit concentration disclosure found.")
            out.append("")

        unallowable = _extract_sentences(
            text, UNALLOWABLE_KEYWORDS,
            max_sentences=EXT_SENTENCES_PER_DIMENSION
        )
        if unallowable:
            out.append("[DISALLOWANCE / COST COMPLIANCE — VERBATIM]")
            for s in unallowable:
                out.append(f"  {s}")
            out.append("")
        else:
            out.append("[DISALLOWANCE / COST COMPLIANCE]")
            out.append("  This document contains no disallowance, cost "
                       "principle, or grantor clawback disclosure. The "
                       "topic is not addressed in this engagement.")
            out.append("")

        opinion = _extract_sentences(
            text, AUDIT_OPINION_KEYWORDS, max_sentences=6,
            exclude=["our objectives are to obtain",
                     "in performing an audit in accordance"]
        )
        if opinion:
            out.append("[AUDIT OPINION — VERBATIM]")
            for s in opinion:
                out.append(f"  {s}")
            out.append("")

        doubt = _extract_sentences(
            text, GOING_CONCERN_DOUBT, max_sentences=6,
            exclude=GOING_CONCERN_BOILERPLATE
        )
        clean = _extract_sentences(
            text, GOING_CONCERN_CLEAN, max_sentences=6,
            exclude=GOING_CONCERN_BOILERPLATE
        )
        out.append("[GOING CONCERN — VERBATIM]")
        if doubt:
            for s in doubt:
                out.append(f"  {s}")
        elif clean:
            for s in clean:
                out.append(f"  {s}")
        else:
            out.append("  No going concern doubt or affirmation "
                       "statement located.")
        out.append("")

        subsequent = _extract_sentences(
            text, SUBSEQUENT_EVENT_KEYWORDS,
            max_sentences=EXT_SENTENCES_PER_DIMENSION,
            exclude=["subsequent events are events or transactions",
                     "have evaluated subsequent events through"]
        )
        if subsequent:
            out.append("[SUBSEQUENT EVENTS — VERBATIM]")
            for s in subsequent:
                out.append(f"  {s}")
            out.append("")

        related = _extract_lines(
            text, RELATED_PARTY_KEYWORDS,
            require_figure=False, max_lines=EXT_LINES_PER_DIMENSION
        )
        if related:
            out.append("[RELATED PARTIES — VERBATIM]")
            for line in related:
                out.append(f"  {line}")
            out.append("")

    out.append("END OF EXTRACT. Base every finding only on the content above.")

    return "\n".join(out).strip()