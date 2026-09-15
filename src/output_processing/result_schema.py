"""
One ExperimentResult is created per document per strategy per run.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class ComplianceFlag:
    """One flag raised by the LLM for a specific compliance dimension."""
    dimension:   str
    severity:    str
    evidence:    str
    is_faithful: Optional[bool] = None   # filled by faithfulness checker
    match_score: Optional[float] = None  # fuzzy match score 0.0 to 1.0


@dataclass
class ExperimentResult:
    """
    Complete result for one API call.
    Stores everything needed for all four evaluation metrics.
    """
    doc_id:        str
    strategy:      str
    run:           int

    # Six compliance dimensions
    revenue_concentration:    str = "CLEAR"
    expense_spike:            str = "CLEAR"
    passthrough_risk:         str = "CLEAR"
    unallowable_expenditure:  str = "CLEAR"
    audit_opinion:            str = "CLEAR"
    going_concern:            str = "CLEAR"
    overall_verdict:          str = "COMPLIANT"
    confidence:               float = 0.0

    flags: List[ComplianceFlag] = field(default_factory=list)

    tokens_input:  int = 0
    tokens_output: int = 0
    tokens_total:  int = 0

    faithfulness_score:    float = 1.0  # proportion of grounded flags
    hallucination_rate:    float = 0.0  # proportion of unsupported flags
    hallucinated_flags:    List[str] = field(default_factory=list)

    # Verbatim quotation vs synthesised claims. A synthesised claim may be
    # factually correct yet untraceable to a single line, which weakens it
    # as audit evidence.
    verbatim_rate:         float = 1.0
    synthesis_rate:        float = 0.0
    mean_match_score:      float = 1.0
    synthesised_flags:     List[str] = field(default_factory=list)

    parse_error:   bool = False
    error_message: str  = ""
    model:         str  = ""
    raw_response:  str  = ""

    def get_dimension_labels(self) -> Dict[str, str]:
        """Returns all six dimension labels as a dict."""
        return {
            "revenue_concentration":   self.revenue_concentration,
            "expense_spike":           self.expense_spike,
            "passthrough_risk":        self.passthrough_risk,
            "unallowable_expenditure": self.unallowable_expenditure,
            "audit_opinion":           self.audit_opinion,
            "going_concern":           self.going_concern,
        }

    def is_flagged(self, dimension: str) -> int:
        """Returns 1 if dimension is FLAG or ESCALATE, 0 if CLEAR or N/A."""
        label = self.get_dimension_labels().get(dimension, "CLEAR")
        return 1 if label in ("FLAG", "ESCALATE") else 0

    def to_dict(self) -> dict:
        """Serialise to dict for JSON storage."""
        return {
            "doc_id":                  self.doc_id,
            "strategy":                self.strategy,
            "run":                     self.run,
            "revenue_concentration":   self.revenue_concentration,
            "expense_spike":           self.expense_spike,
            "passthrough_risk":        self.passthrough_risk,
            "unallowable_expenditure": self.unallowable_expenditure,
            "audit_opinion":           self.audit_opinion,
            "going_concern":           self.going_concern,
            "overall_verdict":         self.overall_verdict,
            "confidence":              self.confidence,
            "flags": [
                {
                    "dimension":   f.dimension,
                    "severity":    f.severity,
                    "evidence":    f.evidence,
                    "is_faithful": f.is_faithful,
                    "match_score": f.match_score,
                }
                for f in self.flags
            ],
            "tokens_input":         self.tokens_input,
            "tokens_output":        self.tokens_output,
            "tokens_total":         self.tokens_total,
            "faithfulness_score":   self.faithfulness_score,
            "hallucination_rate":   self.hallucination_rate,
            "hallucinated_flags":   self.hallucinated_flags,
            "verbatim_rate":        self.verbatim_rate,
            "synthesis_rate":       self.synthesis_rate,
            "mean_match_score":     self.mean_match_score,
            "synthesised_flags":    self.synthesised_flags,
            "parse_error":          self.parse_error,
            "error_message":        self.error_message,
            "model":                self.model,
        }