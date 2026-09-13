"""
verdict_normaliser.py
---------------------
Maps raw LLM JSON output to the standard six-dimension schema.
Handles minor variations in model phrasing and label capitalisation.
"""

from .result_schema import ExperimentResult, ComplianceFlag

VALID_LABELS = {"CLEAR", "FLAG", "ESCALATE", "N/A"}

LABEL_MAP = {
    "none":       "CLEAR",
    "ok":         "CLEAR",
    "pass":       "CLEAR",
    "clean":      "CLEAR",
    "low":        "FLAG",
    "medium":     "FLAG",
    "moderate":   "FLAG",
    "high":       "ESCALATE",
    "critical":   "ESCALATE",
    "na":         "N/A",
    "n/a":        "N/A",
    "not applicable": "N/A",
    "not assessed":   "N/A",
}

DIMENSIONS = [
    "revenue_concentration",
    "expense_spike",
    "passthrough_risk",
    "unallowable_expenditure",
    "audit_opinion",
    "going_concern",
]


def _normalise_label(raw: str) -> str:
    """Normalise a raw LLM label to one of CLEAR / FLAG / ESCALATE / N/A."""
    if not raw:
        return "CLEAR"
    cleaned = str(raw).strip().upper()
    if cleaned in VALID_LABELS:
        return cleaned
    # try lowercase mapping
    lower = str(raw).strip().lower()
    return LABEL_MAP.get(lower, "CLEAR")


def normalise_verdict(
    raw_result: dict,
    doc_id: str,
    strategy: str,
    run: int
) -> ExperimentResult:
    """
    Takes raw LLM API result dict and returns a clean ExperimentResult.

    Args:
        raw_result: output from call_llm_with_retry()
        doc_id:     document identifier
        strategy:   strategy name e.g. S1_full
        run:        run number 0, 1, or 2
    """
    verdict = raw_result.get("verdict", {})
    parse_error = verdict.get("parse_error", False)

    # Build flags list
    flags = []
    for f in verdict.get("flags", []):
        flags.append(ComplianceFlag(
            dimension=f.get("dimension", "unknown"),
            severity=_normalise_label(f.get("severity", "FLAG")),
            evidence=f.get("evidence", ""),
        ))

    result = ExperimentResult(
        doc_id=doc_id,
        strategy=strategy,
        run=run,

        revenue_concentration=_normalise_label(
            verdict.get("revenue_concentration", "CLEAR")),
        expense_spike=_normalise_label(
            verdict.get("expense_spike", "CLEAR")),
        passthrough_risk=_normalise_label(
            verdict.get("passthrough_risk", "CLEAR")),
        unallowable_expenditure=_normalise_label(
            verdict.get("unallowable_expenditure", "CLEAR")),
        audit_opinion=_normalise_label(
            verdict.get("audit_opinion", "CLEAR")),
        going_concern=_normalise_label(
            verdict.get("going_concern", "CLEAR")),
        overall_verdict=verdict.get("overall_verdict", "COMPLIANT"),
        confidence=float(verdict.get("confidence", 0.0)),

        flags=flags,

        tokens_input=raw_result.get("tokens_input", 0),
        tokens_output=raw_result.get("tokens_output", 0),
        tokens_total=raw_result.get("tokens_total", 0),

        parse_error=parse_error,
        error_message=verdict.get("error_message", ""),
        model=raw_result.get("model", ""),
        raw_response=raw_result.get("raw_response", ""),
    )

    return result