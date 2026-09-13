"""
faithfulness_checker.py
-----------------------
Checks whether evidence cited by the LLM in its compliance flags
actually exists in the source document or prepared strategy text.

Two faithfulness modes:
- source: checks against raw PDF text (S1, S2)
- summary: checks against prepared strategy text (S3, S4)

This distinction is itself a methodological contribution — S3 and S4
evidence citations are grounded in the structured summary, not the
raw document, which has different implications for auditability.
"""

from fuzzywuzzy import fuzz
from .result_schema import ExperimentResult, ComplianceFlag

# Threshold: evidence must match at least this well to be grounded
FAITHFULNESS_THRESHOLD = 60  # lowered from 75 — accounts for paraphrasing

# Strategies where evidence should be checked against prepared text
# rather than raw source (because the LLM only sees the summary)
SUMMARY_STRATEGIES = {"S3_fields", "S4_hybrid"}


def _best_match_score(evidence: str, source_text: str) -> float:
    """
    Find the best fuzzy match between cited evidence
    and any sentence in the comparison text.
    Returns a score from 0 to 100.
    """
    if not evidence or not source_text:
        return 0.0

    evidence_lower = evidence.lower().strip()

    sentences = [
        s.strip().lower()
        for s in source_text.replace('\n', '. ').split('.')
        if len(s.strip()) > 5
    ]

    if not sentences:
        return 0.0

    best = max(
        fuzz.partial_ratio(evidence_lower, sentence)
        for sentence in sentences
    )

    return float(best)


def compute_faithfulness(
    result:      ExperimentResult,
    source_text: str,
    prepared_text: str = ""
) -> ExperimentResult:
    """
    For each flag, check whether the cited evidence exists in
    the comparison text. For S3 and S4, checks against the
    prepared strategy text. For S1 and S2, checks against
    the raw source document.

    Args:
        result:        ExperimentResult from verdict_normaliser
        source_text:   full raw PDF text
        prepared_text: text sent to LLM (strategy output)

    Returns:
        Updated ExperimentResult with faithfulness scores filled in
    """
    if not result.flags:
        result.faithfulness_score = 1.0
        result.hallucination_rate = 0.0
        result.hallucinated_flags = []
        return result

    # Choose comparison text based on strategy
    if result.strategy in SUMMARY_STRATEGIES and prepared_text:
        comparison_text = prepared_text
        mode = "summary"
    else:
        comparison_text = source_text
        mode = "source"

    faithful_count = 0
    hallucinated   = []

    for flag in result.flags:
        score = _best_match_score(flag.evidence, comparison_text)
        flag.match_score  = score / 100.0
        flag.is_faithful  = score >= FAITHFULNESS_THRESHOLD

        if flag.is_faithful:
            faithful_count += 1
        else:
            hallucinated.append(
                f"{flag.dimension}: '{flag.evidence}' "
                f"(match score: {score:.0f}, mode: {mode})"
            )

    total = len(result.flags)
    result.faithfulness_score = faithful_count / total if total > 0 else 1.0
    result.hallucination_rate = 1.0 - result.faithfulness_score
    result.hallucinated_flags = hallucinated

    return result