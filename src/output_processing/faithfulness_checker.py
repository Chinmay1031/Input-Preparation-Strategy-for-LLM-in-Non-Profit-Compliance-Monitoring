"""
faithfulness_checker.py
-----------------------
Checks whether evidence cited by the LLM in its compliance flags
actually exists in the source document or prepared strategy text.

Two comparison modes:
  - source:  checks against raw PDF text (S1, S2)
  - summary: checks against prepared strategy text (S3, S4)

This distinction is a methodological contribution. S3 and S4 evidence
is grounded in the structured extract rather than the raw document,
which has different implications for auditability.

Three evidence behaviours are distinguished rather than a single
binary faithful/unfaithful judgement:

  Verbatim   (>= 92)  the model copied a line from the material given
  Synthesis  (60-91)  the model constructed a claim from source content
  Unsupported (< 60)  the claim is not present in the material given

The middle band matters for compliance work. A synthesised claim such
as "expenses increased from X to Y" may be factually correct, yet it
cannot be traced to a single line of the document, which weakens its
value as audit evidence.
"""

from fuzzywuzzy import fuzz
from .result_schema import ExperimentResult, ComplianceFlag

# Evidence is treated as grounded at or above this score
FAITHFULNESS_THRESHOLD = 60

# At or above this score the evidence is a near-exact copy of a
# source line rather than a reconstruction
VERBATIM_THRESHOLD = 92

# Strategies where evidence is checked against the prepared text
# rather than the raw source, because the model only saw the extract
SUMMARY_STRATEGIES = {"S3_fields", "S4_hybrid"}


def _best_match_score(evidence: str, source_text: str) -> float:
    """
    Find the best fuzzy match between cited evidence and any
    sentence in the comparison text. Returns a score from 0 to 100.
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
    result:        ExperimentResult,
    source_text:   str,
    prepared_text: str = ""
) -> ExperimentResult:
    """
    Score each cited flag against the material the model was given,
    and classify the citation behaviour.

    Args:
        result:        ExperimentResult from verdict_normaliser
        source_text:   full raw document text
        prepared_text: text actually sent to the model

    Returns:
        Updated ExperimentResult with faithfulness measures filled in
    """
    if not result.flags:
        result.faithfulness_score = 1.0
        result.hallucination_rate = 0.0
        result.hallucinated_flags = []
        result.verbatim_rate      = 1.0
        result.synthesis_rate     = 0.0
        result.synthesised_flags  = []
        result.mean_match_score   = 1.0
        return result

    # Choose comparison text based on strategy
    if result.strategy in SUMMARY_STRATEGIES and prepared_text:
        comparison_text = prepared_text
        mode = "summary"
    else:
        comparison_text = source_text
        mode = "source"

    faithful_count = 0
    verbatim_count = 0
    synth_count    = 0
    hallucinated   = []
    synthesised    = []
    all_scores     = []

    for flag in result.flags:
        score = _best_match_score(flag.evidence, comparison_text)
        flag.match_score = score / 100.0
        flag.is_faithful = score >= FAITHFULNESS_THRESHOLD
        all_scores.append(score)

        if score >= VERBATIM_THRESHOLD:
            verbatim_count += 1
            faithful_count += 1

        elif score >= FAITHFULNESS_THRESHOLD:
            synth_count    += 1
            faithful_count += 1
            synthesised.append(
                f"{flag.dimension}: '{flag.evidence[:70]}' "
                f"(score {score:.0f}, mode: {mode})"
            )

        else:
            hallucinated.append(
                f"{flag.dimension}: '{flag.evidence[:70]}' "
                f"(score {score:.0f}, mode: {mode})"
            )

    total = len(result.flags)

    result.faithfulness_score = faithful_count / total
    result.hallucination_rate = 1.0 - result.faithfulness_score
    result.hallucinated_flags = hallucinated

    result.verbatim_rate     = verbatim_count / total
    result.synthesis_rate    = synth_count / total
    result.synthesised_flags = synthesised
    result.mean_match_score  = sum(all_scores) / len(all_scores) / 100.0

    return result