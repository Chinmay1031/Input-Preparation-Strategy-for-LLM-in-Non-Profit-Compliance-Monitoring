
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

DOCUMENT_TYPE_SIGNALS = {
    DOCUMENT_TYPE_AUP_REPORT: [
        "AGREED-UPON PROCEDURES",
        "AGREED UPON PROCEDURES",
        "DONATION OF FUNDS AGREEMENT",
        "ISRS 4400",
        "ANNEXURE A",
        "PROCEDURES AND FINDINGS",
    ],
    DOCUMENT_TYPE_FINANCIAL_STATEMENT: [
        "ANNUAL FINANCIAL STATEMENTS",
        "STATEMENT OF FINANCIAL POSITION",
        "STATEMENT OF COMPREHENSIVE INCOME",
        "IFRS FOR SMES",
        "INDEPENDENT REVIEWER",
        "GOING CONCERN",
        "CONSOLIDATED FINANCIAL STATEMENTS",
        "REPORT OF INDEPENDENT AUDITORS",
        "INDEPENDENT AUDITOR'S REPORT",
        "INDEPENDENT AUDITORS' REPORT",
        "ANNUAL REPORT",
        "STATEMENTS OF ACTIVITIES AND CHANGES IN NET ASSETS",
        "STATEMENTS OF ACTIVITIES",
        "STATEMENT OF ACTIVITIES",
        "STATEMENTS OF FUNCTIONAL EXPENSES",
        "STATEMENT OF FUNCTIONAL EXPENSES",
        "NET ASSETS WITHOUT DONOR RESTRICTIONS",
        "NET ASSETS WITH DONOR RESTRICTIONS",
        "GENERALLY ACCEPTED ACCOUNTING PRINCIPLES",
        "INTERNATIONAL PUBLIC SECTOR ACCOUNTING",
        "IPSAS",
        "STATEMENT OF FINANCIAL PERFORMANCE",
        "IFRS ACCOUNTING STANDARDS",
        "STATEMENT OF CHANGES IN EQUITY",
        "NOTES TO THE FINANCIAL STATEMENTS",
        "NOTES TO THE CONSOLIDATED FINANCIAL",
    ],
}

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
        "REPORT OF INDEPENDENT AUDITORS",
        "INDEPENDENT AUDITORS' REPORT",
        "FORVIS MAZARS",
        "ERNST & YOUNG",
        "KPMG",
        "MOSS ADAMS",
        "RSM US",
        "BAKER TILLY",
        "ARTESIAN CPA",
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

COMPLIANCE_RELEVANT_SECTIONS = {
    SEC_AUDIT_REPORT,
    SEC_DIRECTORS_REPORT,
    SEC_FINANCIAL_STATEMENTS,
    SEC_NOTES,
    SEC_DETAILED_INCOME,

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
    body = page_text[200:] if len(page_text) > 200 else page_text
    upper_body = body.upper()
    upper_full = page_text.upper()

    if document_type == DOCUMENT_TYPE_AUP_REPORT:
        section_defs = AUP_REPORT_SECTIONS
        for section_type, keywords in section_defs.items():
            if any(kw in upper_full for kw in keywords):
                return section_type
        return SEC_UNKNOWN

    PRIORITY_ORDER = [
        SEC_DETAILED_INCOME,
        SEC_NOTES,
        SEC_ACCOUNTING_POLICIES,
        SEC_FINANCIAL_STATEMENTS,
        SEC_DIRECTORS_REPORT,
        SEC_DIRECTORS_RESP,
        SEC_AUDIT_REPORT,
        SEC_GENERAL_INFO,
    ]

    for section_type in PRIORITY_ORDER:
        keywords = FINANCIAL_STATEMENT_SECTIONS.get(section_type, [])
        if section_type == SEC_GENERAL_INFO:
            if any(kw in upper_body for kw in keywords):
                return section_type
        else:
            if any(kw in upper_body for kw in keywords) or\
               any(kw in upper_full for kw in keywords[:2]):
                return section_type

    return SEC_UNKNOWN


def is_compliance_relevant(section_type: str) -> bool:
    return section_type in COMPLIANCE_RELEVANT_SECTIONS


def is_boilerplate(section_type: str) -> bool:
    return section_type in BOILERPLATE_SECTIONS
