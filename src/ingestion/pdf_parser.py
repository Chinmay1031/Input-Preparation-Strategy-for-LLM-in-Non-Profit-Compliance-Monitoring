"""
pdf_parser.py
-------------
Main Phase 1 entry point. Parses any HPF grantee document into
a structured ParsedDocument object.

Handles:
- Annual financial statements (IFRS for SMEs format)
- AUP / Agreed-Upon Procedures reports
- Automatically detects document type from content
"""

import re
from pathlib import Path
from typing import Optional

import pdfplumber

from .document_schema import (
    ParsedDocument, DocumentSection, BudgetLine,
    DOCUMENT_TYPE_AUP_REPORT, DOCUMENT_TYPE_FINANCIAL_STATEMENT,
    SEC_UNKNOWN
)
from .section_classifier import (
    detect_document_type, classify_page
)
from .budget_extractor import (
    extract_budget_lines_from_tables,
    extract_budget_lines_from_text
)


def parse_document(pdf_path: str, doc_id: Optional[str] = None) -> ParsedDocument:
    """
    Main entry point for Phase 1.

    Args:
        pdf_path:  path to PDF file
        doc_id:    optional identifier (defaults to filename stem)

    Returns:
        ParsedDocument with all sections, metadata, and structured data
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if doc_id is None:
        doc_id = path.stem

    doc = ParsedDocument(
        doc_id=doc_id,
        source_path=str(path),
    )

    # ── Step 1: extract all text and tables ───────────────────────────────────
    all_text_parts = []
    all_tables     = []
    page_data      = []  # list of (text, tables) per page

    with pdfplumber.open(pdf_path) as pdf:
        doc.total_pages = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages):
            text   = page.extract_text() or ""
            tables = page.extract_tables() or []

            all_text_parts.append(text)
            all_tables.extend(tables)
            page_data.append((page_num + 1, text, tables))

    doc.full_text = "\n".join(all_text_parts)
    if len(doc.full_text.strip()) < 200:
        from .ocr_fallback import ocr_document, OCR_AVAILABLE
        if OCR_AVAILABLE:
            doc.parse_warnings.append(
                "No text layer detected — OCR fallback used. "
                "Text quality may be lower than native extraction."
            )
            ocr_text = ocr_document(pdf_path)
            doc.full_text = ocr_text
            ocr_pages = ocr_text.split("\n\n")
            page_data = [
                (i + 1, page, [])
                for i, page in enumerate(ocr_pages)
            ]
        else:
            doc.parse_warnings.append(
                "No text layer and OCR not available — "
                "document cannot be processed."
            )

    # ── Step 2: detect document type ─────────────────────────────────────────
    doc.document_type = detect_document_type(doc.full_text)

    # ── Step 3: classify pages into sections ─────────────────────────────────
    section_accumulator: dict = {}
    current_section = SEC_UNKNOWN

    for page_num, text, tables in page_data:
        detected = classify_page(text, doc.document_type)

        # Only update current section if we detected something specific
        # This prevents general_info from capturing all pages
        if detected != SEC_UNKNOWN:
            current_section = detected

        # For financial statements: each page likely belongs to ONE section
        # Use the detected section for THIS page, not the accumulating one
        # This gives cleaner section boundaries
        page_section = detected if detected != SEC_UNKNOWN else current_section

        if page_section not in section_accumulator:
            section_accumulator[page_section] = {
                "text":       "",
                "tables":     [],
                "page_start": page_num,
                "page_end":   page_num,
            }

        section_accumulator[page_section]["text"]    += text + "\n"
        section_accumulator[page_section]["page_end"] = page_num
        section_accumulator[page_section]["tables"].extend(tables)

    # ── Step 4: build DocumentSection objects ────────────────────────────────
    for sec_type, data in section_accumulator.items():
        doc.sections[sec_type] = DocumentSection(
            section_type=sec_type,
            page_start=data["page_start"],
            page_end=data["page_end"],
            raw_text=data["text"],
            tables=data["tables"],
        )

    # ── Step 5: extract metadata ──────────────────────────────────────────────
    doc = _extract_metadata(doc)

    # ── Step 6: extract structured financial data ────────────────────────────
    if doc.document_type == DOCUMENT_TYPE_AUP_REPORT:
        doc = _extract_aup_data(doc, all_tables)
    elif doc.document_type == DOCUMENT_TYPE_FINANCIAL_STATEMENT:
        doc = _extract_financial_statement_data(doc, all_tables)

    # ── Step 7: OCR quality check ────────────────────────────────────────────
    doc.ocr_quality_score = _estimate_ocr_quality(doc.full_text)
    if doc.ocr_quality_score < 0.7:
        doc.parse_warnings.append(
            f"Low OCR quality score ({doc.ocr_quality_score:.2f}) — "
            f"document may be scanned. Results may be less reliable."
        )

    return doc


# ── Metadata extraction ───────────────────────────────────────────────────────

def _extract_metadata(doc: ParsedDocument) -> ParsedDocument:
    """Extract grantee name, year, currency, project name."""
    text  = doc.full_text
    upper = text.upper()

    # ── Year ──────────────────────────────────────────────────────────────────
    year_patterns = [
        r'31 DECEMBER (\d{4})',
        r'YEAR ENDED (\d{4})',
        r'FOR THE YEAR (\d{4})',
        r'1 JANUARY (\d{4}) TO 31 DECEMBER \d{4}',
        r'(\d{4}) ACTUAL',
    ]
    for pattern in year_patterns:
        m = re.search(pattern, upper)
        if m:
            doc.fiscal_year = int(m.group(1))
            break

    # ── Currency ──────────────────────────────────────────────────────────────
    if "FIGURES IN EURO" in upper or "SPAC NPC (EURO)" in upper or "IN EURO" in upper:
        doc.currency = "EUR"
    elif "ZAR" in upper or "SOUTH AFRICAN RAND" in upper:
        doc.currency = "ZAR"
    elif "USD" in upper or "US DOLLAR" in upper:
        doc.currency = "USD"
    else:
        doc.currency = "EUR"  # default for HPF documents

    # ── Grantee name ─────────────────────────────────────────────────────────
    # Look for NPC / Foundation / Charities in first 500 chars
    name_patterns = [
        r'([A-Z][A-Z\s]+(?:NPC|FOUNDATION|CHARITIES|ORGANISATION|TRUST))',
        r'(?:DIRECTORS OF|TO THE DIRECTORS OF)\s+([A-Z][A-Z\s]+)',
    ]
    for pattern in name_patterns:
        m = re.search(pattern, text[:1000].upper())
        if m:
            candidate = m.group(1).strip()
            if 5 < len(candidate) < 80:
                doc.grantee_name = candidate.title()
                break

    # ── Auditor ───────────────────────────────────────────────────────────────
    if "FORVIS MAZARS" in upper:
        doc.auditor = "Forvis Mazars"
    elif "MAZARS" in upper:
        doc.auditor = "Mazars"
    elif "KPMG" in upper:
        doc.auditor = "KPMG"
    elif "PWC" in upper or "PRICEWATERHOUSECOOPERS" in upper:
        doc.auditor = "PwC"

    # ── Project name (AUP specific) ───────────────────────────────────────────
    project_match = re.search(
        r'(?:THE\s+)?([A-Z][A-Z\s]+(?:PROJECT|PROGRAMME|PROGRAM))',
        text[:2000].upper()
    )
    if project_match:
        doc.project_name = project_match.group(1).strip().title()

    return doc


# ── AUP-specific data extraction ──────────────────────────────────────────────

def _extract_aup_data(doc: ParsedDocument, all_tables: list) -> ParsedDocument:
    """Extract structured data specific to AUP reports."""

    # Extract budget lines from Annexure A tables
    budget_lines = extract_budget_lines_from_tables(all_tables)

    # fallback: try text extraction if table extraction yielded little
    if len(budget_lines) < 5:
        from .document_schema import SEC_AUP_ANNEXURE_A
        annexure_text = doc.get_section_text(SEC_AUP_ANNEXURE_A)
        if annexure_text:
            budget_lines = extract_budget_lines_from_text(annexure_text)
            doc.parse_warnings.append(
                "Budget lines extracted from text fallback "
                "(table extraction yielded insufficient results)"
            )

    doc.budget_lines = budget_lines

    # Extract key AUP summary figures
    text = doc.full_text

    # Donations received — look for large numbers after "Donations" keyword
    # Pattern: "Donations 1,800,000 2,400,000" → take first large number
    donation_match = re.search(
        r'Donations\s+([\d,]+)', text, re.IGNORECASE
    )
    if donation_match:
        amount_str = donation_match.group(1).replace(',', '')
        try:
            val = float(amount_str)
            if val > 1000:  # sanity check — must be a meaningful amount
                doc.financial_figures["donations_received"] = val
        except ValueError:
            pass

    # Total expenses — look for the summary line with large number
    expense_match = re.search(
        r'Total Operating Expenses\s+([\d,]+)', text, re.IGNORECASE
    )
    if expense_match:
        amount_str = expense_match.group(1).replace(',', '')
        try:
            val = float(amount_str)
            if val > 1000:
                doc.financial_figures["total_expenses"] = val
        except ValueError:
            pass

    # Closing balance
    closing_match = re.search(
        r'Closing Balance\s+([\d,]+)', text, re.IGNORECASE
    )
    if closing_match:
        amount_str = closing_match.group(1).replace(',', '')
        try:
            val = float(amount_str)
            if val > 1000:
                doc.financial_figures["closing_balance"] = val
        except ValueError:
            pass

    # Extract procedure findings count — indicator of assurance scope
    procedure_count = len(re.findall(r'^\d+\.\s+', text, re.MULTILINE))
    if procedure_count > 0:
        doc.financial_figures["procedure_count"] = procedure_count

    # Check for "no exceptions" — clean audit indicator
    no_exceptions_count = text.upper().count("NO EXCEPTIONS")
    doc.financial_figures["no_exceptions_count"] = no_exceptions_count

    return doc


# ── Financial statement data extraction ───────────────────────────────────────

def _extract_financial_statement_data(doc: ParsedDocument, all_tables: list) -> ParsedDocument:
    """Extract structured data from annual financial statements."""

    # Extract key financial figures using table parsing + regex
    for table in all_tables:
        if not table:
            continue
        for row in table:
            if not row:
                continue
            cells = [str(c).strip() if c else "" for c in row]
            row_text = " ".join(cells).lower()
            numbers = _extract_numbers_from_row(cells)

            if "donation" in row_text or "revenue" in row_text:
                if len(numbers) >= 2:
                    doc.financial_figures["revenue_current"] = numbers[0]
                    doc.financial_figures["revenue_prior"]   = numbers[1]

            if "employee cost" in row_text:
                if len(numbers) >= 2:
                    doc.financial_figures["employee_costs_current"] = numbers[0]
                    doc.financial_figures["employee_costs_prior"]   = numbers[1]

            if "staff welfare" in row_text:
                if len(numbers) >= 2:
                    doc.financial_figures["staff_welfare_current"] = numbers[0]
                    doc.financial_figures["staff_welfare_prior"]   = numbers[1]

            if "profit for the year" in row_text or "total comprehensive" in row_text:
                if len(numbers) >= 2:
                    doc.financial_figures["profit_current"] = numbers[0]
                    doc.financial_figures["profit_prior"]   = numbers[1]

            if "cash and cash equivalent" in row_text:
                if len(numbers) >= 2:
                    doc.financial_figures["cash_current"] = numbers[0]
                    doc.financial_figures["cash_prior"]   = numbers[1]

            if "donations" in row_text and "paid" in row_text:
                if numbers:
                    doc.financial_figures["donations_paid"] = numbers[0]

    return doc


# ── Utility functions ─────────────────────────────────────────────────────────

def _extract_numbers_from_row(cells: list) -> list:
    """Extract numeric values from a list of cell strings."""
    numbers = []
    for cell in cells:
        cleaned = re.sub(r'[€$£,\s]', '', str(cell))
        if cleaned.startswith('(') and cleaned.endswith(')'):
            cleaned = '-' + cleaned[1:-1]
        try:
            n = float(cleaned)
            if abs(n) > 0:  # skip zeros
                numbers.append(n)
        except ValueError:
            continue
    return numbers


def _estimate_ocr_quality(text: str) -> float:
    """
    Estimate OCR quality based on text characteristics.
    Returns score 0.0 (very poor) to 1.0 (clean digital PDF).
    """
    if not text or len(text) < 100:
        return 0.0

    # indicators of clean digital text
    words = text.split()
    if not words:
        return 0.0

    # ratio of recognisable words
    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    quality = alpha_words / len(words)

    # penalise excessive special characters (OCR artifact indicator)
    special_ratio = sum(
        1 for c in text if c in '|~`^<>{}\\@#$%*'
    ) / max(len(text), 1)
    quality -= special_ratio * 2

    return max(0.0, min(1.0, quality))