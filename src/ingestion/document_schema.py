
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


DOCUMENT_TYPE_FINANCIAL_STATEMENT = "financial_statement"
DOCUMENT_TYPE_AUP_REPORT          = "aup_report"
DOCUMENT_TYPE_AUDIT_REPORT         = "audit_report"
DOCUMENT_TYPE_NARRATIVE_REPORT     = "narrative_report"
DOCUMENT_TYPE_UNKNOWN              = "unknown"


SEC_GENERAL_INFO          = "general_info"
SEC_DIRECTORS_RESP        = "directors_responsibilities"
SEC_AUDIT_REPORT          = "audit_report"
SEC_DIRECTORS_REPORT      = "directors_report"
SEC_FINANCIAL_STATEMENTS  = "financial_statements"
SEC_ACCOUNTING_POLICIES   = "accounting_policies"
SEC_NOTES                 = "notes"
SEC_DETAILED_INCOME       = "detailed_income"

SEC_AUP_PURPOSE           = "aup_purpose"
SEC_AUP_PROCEDURES        = "aup_procedures_findings"
SEC_AUP_ANNEXURE_A        = "aup_annexure_a"
SEC_AUP_ANNEXURE_B        = "aup_annexure_b"
SEC_AUP_NOTES             = "aup_notes"

SEC_UNKNOWN               = "unknown"


@dataclass
class BudgetLine:
    category: str
    actual: float
    budget: float
    variance: float
    variance_pct: Optional[float] = None
    comment: Optional[str] = None
    country: Optional[str] = None
    is_overspend: bool = False
    is_unbudgeted: bool = False


@dataclass
class DocumentSection:
    section_type: str
    page_start: int
    page_end: int
    raw_text: str
    tables: List[List] = field(default_factory=list)
    token_estimate: int = 0


@dataclass
class ParsedDocument:
    doc_id: str
    source_path: str
    document_type: str = DOCUMENT_TYPE_UNKNOWN

    grantee_name: str = ""
    fiscal_year: Optional[int] = None
    currency: Optional[str] = None
    project_name: Optional[str] = None
    auditor: Optional[str] = None
    report_date: Optional[str] = None

    sections: Dict[str, DocumentSection] = field(default_factory=dict)
    full_text: str = ""
    total_pages: int = 0

    budget_lines: List[BudgetLine] = field(default_factory=list)
    financial_figures: Dict[str, Any] = field(default_factory=dict)

    parse_warnings: List[str] = field(default_factory=list)
    ocr_quality_score: float = 1.0

    def get_section_text(self, section_type: str) -> str:
        s = self.sections.get(section_type)
        return s.raw_text if s else ""

    def has_section(self, section_type: str) -> bool:
        return section_type in self.sections

    def get_all_compliance_sections_text(self) -> str:
        from .section_classifier import COMPLIANCE_RELEVANT_SECTIONS
        parts = []
        for sec_type, section in self.sections.items():
            if sec_type in COMPLIANCE_RELEVANT_SECTIONS:
                parts.append(f"=== {sec_type.upper()} ===\n{section.raw_text}")
        return "\n\n".join(parts)

    def summary(self) -> str:
        return (
            f"DocID: {self.doc_id} | Type: {self.document_type} | "
            f"Grantee: {self.grantee_name} | Year: {self.fiscal_year} | "
            f"Pages: {self.total_pages} | Sections: {list(self.sections.keys())} | "
            f"Budget lines: {len(self.budget_lines)} | "
            f"Warnings: {len(self.parse_warnings)}"
        )
