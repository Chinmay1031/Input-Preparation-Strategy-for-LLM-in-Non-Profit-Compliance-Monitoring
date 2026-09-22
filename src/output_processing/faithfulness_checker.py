
import re
from fuzzywuzzy import fuzz
from .result_schema import ExperimentResult, ComplianceFlag

FAITHFULNESS_THRESHOLD = 60

VERBATIM_THRESHOLD = 92

FIGURE_MATCH_SCORE = 85.0

SUMMARY_STRATEGIES = {"S3_fields", "S4_hybrid", "S5_extended"}


def _extract_figures(text: str) -> set:
    cleaned = re.sub(r'[$€£()]', ' ', text)
    figures = set()

    for token in cleaned.split():
        digits = re.sub(r'[^\d]', '', token)
        if len(digits) >= 4:
            figures.add(digits)

    return figures


def _best_match_score(evidence: str, source_text: str) -> float:
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

    text_score = max(
        fuzz.partial_ratio(evidence_lower, sentence)
        for sentence in sentences
    )

    cited_figures = _extract_figures(evidence)
    if cited_figures:
        source_figures = _extract_figures(source_text)
        matched = cited_figures & source_figures
        if matched and len(matched) == len(cited_figures):
            return float(max(text_score, FIGURE_MATCH_SCORE))

    return float(text_score)


def compute_faithfulness(
    result:        ExperimentResult,
    source_text:   str,
    prepared_text: str = ""
) -> ExperimentResult:
    if not result.flags:
        result.faithfulness_score = 1.0
        result.hallucination_rate = 0.0
        result.hallucinated_flags = []
        result.verbatim_rate      = 1.0
        result.synthesis_rate     = 0.0
        result.synthesised_flags  = []
        result.mean_match_score   = 1.0
        return result

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