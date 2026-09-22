
from typing import List, Dict
from sklearn.metrics import cohen_kappa_score
from src.output_processing.result_schema import ExperimentResult
from .gold_standard_loader import DIMENSIONS


def compute_consistency_metrics(
    results: List[ExperimentResult]
) -> Dict:
    metrics = {}
    strategies = set(r.strategy for r in results)

    for strategy in strategies:
        runs = {}
        for run_num in range(3):
            run_results = [
                r for r in results
                if r.strategy == strategy and r.run == run_num
            ]
            if run_results:
                runs[run_num] = run_results

        if len(runs) < 2:
            metrics[strategy] = {
                "kappa":             None,
                "kappa_interpretation": "Insufficient runs",
                "agreement_rate":    None,
            }
            continue

        def get_labels(run_num):
            labels = []
            for r in sorted(runs[run_num], key=lambda x: x.doc_id):
                for dim in DIMENSIONS:
                    label = r.get_dimension_labels().get(dim, "CLEAR")
                    if label == "N/A":
                        label = "CLEAR"
                    labels.append(1 if label in ("FLAG", "ESCALATE") else 0)
            return labels

        kappas = []
        agreements = []

        run_pairs = [
            (0, 1), (0, 2), (1, 2)
        ]
        for r1, r2 in run_pairs:
            if r1 in runs and r2 in runs:
                labels_r1 = get_labels(r1)
                labels_r2 = get_labels(r2)

                if len(labels_r1) == len(labels_r2) and len(labels_r1) > 0:
                    if len(set(labels_r1 + labels_r2)) == 1:
                        kappas.append(1.0)
                    else:
                        try:
                            k = cohen_kappa_score(labels_r1, labels_r2)
                            kappas.append(k)
                        except Exception:
                            kappas.append(0.0)

                    agree = sum(
                        1 for a, b in zip(labels_r1, labels_r2)
                        if a == b
                    ) / len(labels_r1)
                    agreements.append(agree)

        if kappas:
            avg_kappa = sum(kappas) / len(kappas)
            avg_agree = sum(agreements) / len(agreements)

            metrics[strategy] = {
                "kappa":             round(avg_kappa, 3),
                "kappa_interpretation": _interpret_kappa(avg_kappa),
                "agreement_rate":    round(avg_agree, 3),
                "kappa_pairs":       {
                    f"run{r1}_vs_run{r2}": round(k, 3)
                    for (r1, r2), k in zip(run_pairs, kappas)
                },
            }

    return metrics


def _interpret_kappa(kappa: float) -> str:
    if kappa >= 0.81: return "Almost perfect"
    if kappa >= 0.61: return "Substantial"
    if kappa >= 0.41: return "Moderate"
    if kappa >= 0.21: return "Fair"
    return "Slight"