
import re
from src.ingestion.document_schema import (
    ParsedDocument,
    DOCUMENT_TYPE_AUP_REPORT,
    DOCUMENT_TYPE_FINANCIAL_STATEMENT,
)
from src.ingestion.budget_extractor import summarise_budget_lines


MAX_LINES_PER_DIMENSION = 6
MIN_LINE_LENGTH         = 8
MAX_LINE_LENGTH         = 200

NOISE_PATTERNS = [
    r'^\s*page\s+\d+',
    r'^\s*\d+\s*$',
    r'see accompanying notes',
    r'the accompanying notes are an integral',
    r'^\s*table of contents',
    r'\.{6,}',
]


def _is_noise(line: str) -> bool:
    low = line.lower()
    return any(re.search(p, low) for p in NOISE_PATTERNS)


def _has_figure(line: str) -> bool:
    return bool(re.search(r'\d[\d,\. ]{2,}', line))


def _clean(line: str) -> str:
    return re.sub(r'\s{2,}', '  ', line.strip())


def _extract_lines(
    text: str,
    keywords: list,
    require_figure: bool = True,
    max_lines: int = MAX_LINES_PER_DIMENSION,
    exclude: list = None,
) -> list:
    exclude = exclude or []
    found   = []
    seen    = set()

    for raw in text.split('\n'):
        line = _clean(raw)
        low  = line.lower()

        if len(line) < MIN_LINE_LENGTH or len(line) > MAX_LINE_LENGTH:
            continue
        if _is_noise(line):
            continue
        if any(x in low for x in exclude):
            continue
        if not any(k in low for k in keywords):
            continue
        if require_figure and not _has_figure(line):
            continue
        if low in seen:
            continue

        seen.add(low)
        found.append(line)

        if len(found) >= max_lines:
            break

    return found


def _extract_sentences(
    text: str,
    keywords: list,
    max_sentences: int = 3,
    exclude: list = None,
) -> list:
    exclude   = exclude or []
    sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
    found     = []

    for s in sentences:
        s   = _clean(s)
        low = s.lower()

        if len(s) < 40 or len(s) > 400:
            continue
        if any(x in low for x in exclude):
            continue
        if not any(k in low for k in keywords):
            continue

        found.append(s)
        if len(found) >= max_sentences:
            break

    return found


REVENUE_KEYWORDS = [
    "total revenue", "total support", "total income",
    "contributions", "donations received", "grant income",
    "revenue from", "total revenues",
]

EXPENSE_KEYWORDS = [
    "total expenses", "total operating expenses", "total expense",
    "personnel costs", "salaries and", "staff costs", "employee costs",
    "total salaries", "occupancy expense", "travel expense",
    "professional fees", "total functional expenses",
]

PASSTHROUGH_KEYWORDS = [
    "grants/subgrants", "sub-grant", "subgrant",
    "grants to", "grants awarded", "grants payable",
    "grants and charges", "pass-through", "sub-recipient",
    "subrecipient", "fund manager",
]

CONCENTRATION_KEYWORDS = [
    "concentration", "significant donor", "significant funder",
    "major donor", "% of total support", "% of total revenue",
    "of total contributions", "depends on continuous funding",
    "dependent upon continued funding",
]

UNALLOWABLE_KEYWORDS = [
    "disallow", "cost principles", "2 cfr 200",
    "uniform guidance", "questioned cost", "unallowable",
    "subject to audit by grantor", "reimbursement from the organization",
]

GOING_CONCERN_DOUBT = [
    "substantial doubt about the",
    "material uncertainty related to going concern",
    "raise substantial doubt about its ability",
    "ability to continue as a going concern is dependent",
]

GOING_CONCERN_CLEAN = [
    "adequate resources to continue", "adequate financial resources",
    "no material uncertainty", "sound financial position",
    "not aware of any material",
]

GOING_CONCERN_BOILERPLATE = [
    "our objectives are to obtain",
    "conclude whether, in our judgment",
    "in performing an audit in accordance",
    "auditor's responsibilities for the audit",
    "management is required to evaluate whether",
]

SUBSEQUENT_EVENT_KEYWORDS = [
    "executive order", "terminated", "reduction in force",
    "funding disruption", "awards have resumed",
    "situation continues to evolve", "suspension of",
]

AUDIT_OPINION_KEYWORDS = [
    "in our opinion", "present fairly, in all material respects",
    "true and fair view", "nothing has come to our attention",
    "unmodified opinion", "qualified opinion", "adverse opinion",
    "we do not express an opinion",
]

RELATED_PARTY_KEYWORDS = [
    "related part", "due from related", "due to related",
    "common interest", "wholly owned subsidiary",
    "member of the same", "affiliate",
]
NOISE_PATTERNS = [
    r'^\s*page\s+\d+',
    r'^\s*\d+\s*$',
    r'see accompanying notes',
    r'the accompanying notes are an integral',
    r'^\s*table of contents',
    r'\.{6,}',
    r'^\s*[•*=■▪]',                                              
    r'participants have',                              
    r'\b(farmers|beneficiaries|households|villages)\b',
]


def prepare_s3(doc: ParsedDocument) -> str:
    out  = []
    text = doc.full_text

    out.append("GRANTEE COMPLIANCE EXTRACT")
    out.append(f"Grantee:       {doc.grantee_name or 'Not identified'}")
    out.append(f"Document type: {doc.document_type or 'Unknown'}")
    out.append(f"Fiscal year:   {doc.fiscal_year or 'Not identified'}")
    out.append(f"Auditor:       {doc.auditor or 'Not identified'}")
    out.append("")
    out.append("The following lines are extracted verbatim from the source "
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

        findings = _extract_lines(
            doc.get_section_text("aup_purpose") + doc.full_text,
            ["exception", "difference", "variance", "overspend",
             "falls out the scope", "reallocat"],
            require_figure=False,
            max_lines=6,
        )
        if findings:
            out.append("[AUDITOR FINDINGS — VERBATIM]")
            for f in findings:
                out.append(f"  {f}")
            out.append("")

    else:
        revenue = _extract_lines(text, REVENUE_KEYWORDS, max_lines=5)
        if revenue:
            out.append("[REVENUE AND SUPPORT — VERBATIM]")
            for line in revenue:
                out.append(f"  {line}")
            out.append("")

        expenses = _extract_lines(text, EXPENSE_KEYWORDS, max_lines=6)
        if expenses:
            out.append("[EXPENSES — VERBATIM]")
            for line in expenses:
                out.append(f"  {line}")
            out.append("")

        passthrough = _extract_lines(
            text, PASSTHROUGH_KEYWORDS, require_figure=False, max_lines=5
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
            text, CONCENTRATION_KEYWORDS, max_sentences=4
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
            text, UNALLOWABLE_KEYWORDS, max_sentences=3
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

        audit_src = (doc.get_section_text("audit_report") or text[:12000])
        opinion = _extract_sentences(
            audit_src, AUDIT_OPINION_KEYWORDS, max_sentences=2,
            exclude=["our objectives are to obtain",
                     "in performing an audit in accordance"]
        )
        if opinion:
            out.append("[AUDIT OPINION — VERBATIM]")
            for s in opinion:
                out.append(f"  {s}")
            out.append("")

        doubt = _extract_sentences(
            text, GOING_CONCERN_DOUBT, max_sentences=2,
            exclude=GOING_CONCERN_BOILERPLATE
        )
        clean = _extract_sentences(
            text, GOING_CONCERN_CLEAN, max_sentences=2,
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
            text, SUBSEQUENT_EVENT_KEYWORDS, max_sentences=3,
            exclude=["subsequent events are events or transactions",
                     "have evaluated subsequent events through"]
        )
        if subsequent:
            out.append("[SUBSEQUENT EVENTS — VERBATIM]")
            for s in subsequent:
                out.append(f"  {s}")
            out.append("")

        related = _extract_lines(
            text, RELATED_PARTY_KEYWORDS, require_figure=False, max_lines=4
        )
        if related:
            out.append("[RELATED PARTIES — VERBATIM]")
            for line in related:
                out.append(f"  {line}")
            out.append("")

    out.append("END OF EXTRACT. Base every finding only on the lines above.")

    return "\n".join(out).strip()