"""
section_classifier.py
---------------------
Classifies each page of a document into a section type.
Handles BOTH financial statements AND AUP reports.

This domain-specific classification is a core technical contribution —
it encodes knowledge of nonprofit financial document structure
that no generic PDF parser possesses.
"""

from .document_schema import (
    DOCUMENT_TYPE_FINANCIAL_STATEMENT,
    DOCUMENT_TYPE_AUP_REPORT,
    DOCUMENT_TYPE_UNKNOWN,
    SEC_GENERAL_INFO, SEC_DIRECTORS_RESP, SEC_AUDIT_REPORT,
    SEC_DIRECTORS_REPORT, SEC_FINANCIAL_STATEMENTS,
    SEC_ACCOUNTING_POLICIES, SEC_NOTES, SEC_DETAILED_INCOME,
    SEC_AUP_PURPOSE, SEC_AUP_PROCEDURES, SEC_AUP_ANNEXURE_A,
    SEC_AUP_ANNEXURE_B, SEC_AUP_NOTES, SEC_UNKNOWN
)

# ── Document type detection keywords ─────────────────────────────────────────
DOCUMENT_TYPE_SIGNALS = {
    DOCUMENT_TYPE_AUP_REPORT: [
        "AGREED-UPON PROCEDURES",
        "AGREED UPON PROCEDURES",
        "DONATION OF FUNDS AGREEMENT",
        "ISRS 4400",
        "ANNEXURE A",
        "BUDGET",
        "VARIANCE",
    ],
    DOCUMENT_TYPE_FINANCIAL_STATEMENT: [
        "ANNUAL FINANCIAL STATEMENTS",
        "STATEMENT OF FINANCIAL POSITION",
        "STATEMENT OF COMPREHENSIVE INCOME",
        "IFRS FOR SMES",
        "INDEPENDENT REVIEWER",
        "GOING CONCERN",
    ],
}

# ── Section keywords — financial statement ────────────────────────────────────
FINANCIAL_STATEMENT_SECTIONS = {
    SEC_GENERAL_INFO: [
        "GENERAL INFORMATION",
        "COUNTRY OF INCORPORATION",
        "REGISTRATION NUMBER",
        "NATURE OF BUSINESS",
        "REGISTERED OFFICE",
    ],
    SEC_DIRECTORS_RESP: [
        "DIRECTORS' RESPONSIBILITIES",
        "DIRECTORS RESPONSIBILITIES AND APPROVAL",
        "RESPONSIBILITIES AND APPROVAL",
    ],
    SEC_AUDIT_REPORT: [
        "INDEPENDENT REVIEWER'S REPORT",
        "INDEPENDENT REVIEWER",
        "INDEPENDENT AUDITOR",
        "FORVIS MAZARS",
    ],
    SEC_DIRECTORS_REPORT: [
        "DIRECTORS' REPORT",
        "DIRECTORS REPORT",
        "REVIEW OF FINANCIAL RESULTS",
        "NATURE OF BUSINESS",
        "GOING CONCERN",
        "EVENTS AFTER THE REPORTING",
    ],
    SEC_FINANCIAL_STATEMENTS: [
        "STATEMENT OF FINANCIAL POSITION",
        "STATEMENT OF COMPREHENSIVE INCOME",
        "STATEMENT OF CASH FLOWS",
        "STATEMENT OF CHANGES IN EQUITY",
        "TOTAL ASSETS",
        "RETAINED INCOME",
        "FIGURES IN EURO",
    ],
    SEC_ACCOUNTING_POLICIES: [
        "ACCOUNTING POLICIES",
        "BASIS OF PREPARATION",
        "SIGNIFICANT JUDGEMENTS",
        "SUMMARY OF SIGNIFICANT ACCOUNTING",
        "INVESTMENT PROPERTY",
        "PROPERTY, PLANT AND EQUIPMENT",
    ],
    SEC_NOTES: [
        "NOTES TO THE ANNUAL FINANCIAL",
        "NOTES TO THE FINANCIAL STATEMENTS",
        "INVESTMENT PROPERTY",
        "TRADE AND OTHER RECEIVABLES",
        "RELATED PARTIES",
        "DIRECTORS' REMUNERATION",
    ],
    SEC_DETAILED_INCOME: [
        "DETAILED INCOME STATEMENT",
        "ACCOUNTING FEES",
        "ADMINISTRATION FEES",
        "BANK CHARGES",
        "EMPLOYEE COSTS",
        "TEACHER TRAINING",
    ],
}

# ── Section keywords — AUP report ────────────────────────────────────────────
AUP_REPORT_SECTIONS = {
    SEC_AUP_PURPOSE: [
        "PURPOSE OF THIS AGREED",
        "PURPOSE OF THE AGREED",
        "RESPONSIBILITIES OF THE ENGAGING",
        "PRACTITIONERS RESPONSIBILITIES",
        "PROFESSIONAL ETHICS",
    ],
    SEC_AUP_PROCEDURES: [
        "PROCEDURES AND FINDINGS",
        "NO. PROCEDURE",
        "PROCEDURE FINDINGS",
    ],
    SEC_AUP_ANNEXURE_A: [
        "ANNEXURE A",
        "EDUCONSERVATION",
        "OPENING BALANCE",
        "TOTAL INCOME",
        "OPERATING EXPENSES",
        "DEVELOPMENT EXPENSES",
    ],
    SEC_AUP_ANNEXURE_B: [
        "ANNEXURE B",
        "CATEGORY PER ANNEXURE A",
        "AMOUNT PER GENERAL LEDGER",
        "DESCRIPTION PER INVOICE",
        "PETTY CASH",
    ],
    SEC_AUP_NOTES: [
        "NOTES\n",
        "1C ROUNDING",
        "PER ANN JACOBS",
        "INVOICES THAT WERE NOT IN EUROS",
    ],
}

# ── Compliance-relevant sections per document type ───────────────────────────
# S2 strategy uses this to filter out boilerplate
COMPLIANCE_RELEVANT_SECTIONS = {
    # Financial statement — compliance signal sections
    SEC_AUDIT_REPORT,
    SEC_DIRECTORS_REPORT,
    SEC_FINANCIAL_STATEMENTS,
    SEC_NOTES,
    SEC_DETAILED_INCOME,

    # AUP report — ALL sections are compliance relevant
    SEC_AUP_PURPOSE,
    SEC_AUP_PROCEDURES,
    SEC_AUP_ANNEXURE_A,
    SEC_AUP_ANNEXURE_B,
    SEC_AUP_NOTES,
}

BOILERPLATE_SECTIONS = {
    SEC_GENERAL_INFO,
    SEC_DIRECTORS_RESP,
    SEC_ACCOUNTING_POLICIES,
}


def detect_document_type(full_text: str) -> str:
    """
    Determine whether this is a financial statement or AUP report.
    Checks first 2000 characters (covers page 1 reliably).
    """
    sample = full_text[:3000].upper()
    scores = {doc_type: 0 for doc_type in DOCUMENT_TYPE_SIGNALS}

    for doc_type, keywords in DOCUMENT_TYPE_SIGNALS.items():
        for kw in keywords:
            if kw in sample:
                scores[doc_type] += 1

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return DOCUMENT_TYPE_UNKNOWN
    return best


def classify_page(page_text: str, document_type: str) -> str:
    """
    Classify a single page into a section type.
    Uses priority ordering — more specific sections checked first.
    Checks body content (after first 200 chars) to avoid header pollution.
    """
    # Use body text — skip the repeated header (first ~200 chars of each page)
    # The header "SABINE PLATTNER... Annual Financial Statements..." repeats on every page
    body = page_text[200:] if len(page_text) > 200 else page_text
    upper_body = body.upper()
    upper_full = page_text.upper()

    if document_type == DOCUMENT_TYPE_AUP_REPORT:
        section_defs = AUP_REPORT_SECTIONS
        # For AUP check full page (no repeated header issue)
        for section_type, keywords in section_defs.items():
            if any(kw in upper_full for kw in keywords):
                return section_type
        return SEC_UNKNOWN

    # Financial statement — use priority ordering to avoid general_info
    # winning on every page due to "REGISTRATION NUMBER" in header
    PRIORITY_ORDER = [
        SEC_DETAILED_INCOME,      # most specific — check first
        SEC_NOTES,
        SEC_ACCOUNTING_POLICIES,
        SEC_FINANCIAL_STATEMENTS,
        SEC_DIRECTORS_REPORT,
        SEC_DIRECTORS_RESP,
        SEC_AUDIT_REPORT,
        SEC_GENERAL_INFO,         # least specific — check last
    ]

    for section_type in PRIORITY_ORDER:
        keywords = FINANCIAL_STATEMENT_SECTIONS.get(section_type, [])
        # Check body first (avoids header pollution)
        # For general_info specifically require keywords in body too
        if section_type == SEC_GENERAL_INFO:
            if any(kw in upper_body for kw in keywords):
                return section_type
        else:
            if any(kw in upper_body for kw in keywords) or \
               any(kw in upper_full for kw in keywords[:2]):  # first 2 kw checked in full
                return section_type

    return SEC_UNKNOWN


def is_compliance_relevant(section_type: str) -> bool:
    return section_type in COMPLIANCE_RELEVANT_SECTIONS


def is_boilerplate(section_type: str) -> bool:
    return section_type in BOILERPLATE_SECTIONS