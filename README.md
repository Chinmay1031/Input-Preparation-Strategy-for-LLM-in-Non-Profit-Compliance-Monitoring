# Input Strategies for Large Language Models in Non-Profit Compliance Monitoring

> **Master's Thesis Project** — An empirical comparison of five document-preparation strategies for LLM-based compliance review of nonprofit grantee financial statements.

**Author:** Chinmay Bandekar
**Model under study:** OpenAI GPT-4o
**Domain:** Foundation grantee compliance / nonprofit financial oversight

---

## 1. Motivation

Foundations that award grants to nonprofits are expected to monitor how those funds are used. In practice, a programme officer may be responsible for reviewing dozens of grantee audit reports and financial statements per year — long, unstructured PDFs of 30–150+ pages each. A full read is expensive; skimming risks missing material compliance issues (revenue concentration, unallowable expenditure, going-concern doubt, adverse audit opinions, etc.).

Large language models are an obvious candidate for triage, but naively feeding an entire audited financial statement into a model is:

- **Expensive** — long contexts cost more per document and scale poorly across a grant portfolio.
- **Noisy** — most of the document is boilerplate; the compliance-relevant material is a small fraction of the text.
- **Bounded** — some documents exceed the model's practical context window and must be truncated.

This thesis asks: **how does the way we prepare a document as LLM input affect the quality, consistency, faithfulness, and cost of the compliance verdict the model returns?**

## 2. Research question

> *For automated compliance monitoring of nonprofit financial documents, how does the choice of input preparation strategy affect an LLM's classification quality, run-to-run consistency, faithfulness to source evidence, and per-document cost?*

The study compares five input strategies (§3) across six compliance dimensions (§4) on a set of publicly available audited nonprofit financial statements and Agreed-Upon-Procedures (AUP) reports (§5), scored against a manually constructed gold standard.

## 3. The five input strategies

Each strategy takes the same parsed document and prepares a different text payload for the same prompt and model. Only the input text changes between conditions; the model, temperature, prompt, and output schema are held constant.

| ID  | Name               | What it sends                                                              | Design intent                                                 |
| --- | ------------------ | -------------------------------------------------------------------------- | ------------------------------------------------------------- |
| S1  | Full text          | The whole document verbatim (truncated only if it exceeds the input cap)   | Ceiling condition — maximum information, maximum cost         |
| S2  | Section filter     | Only sections classified as compliance-relevant (audit report, notes, etc.) | Removes boilerplate while keeping full section context        |
| S3  | Field extraction   | Compact keyword-anchored line extracts per compliance dimension            | Aggressive compression — minimum tokens                        |
| S4  | Hybrid             | Field extracts plus a small budget summary and key audit sentences         | Balance between S2's context and S3's compression             |
| S5  | Extended extraction | Full compliance-critical sections plus expanded field extracts             | "Rich extract" — cheaper than S1 but retains raw section text |

Strategy implementations live under [src/strategies/](src/strategies/).

## 4. Compliance dimensions scored

Every run returns a structured JSON verdict labelling the document across six dimensions, each as `CLEAR`, `FLAG`, or `ESCALATE` (with `N/A` permitted for going concern where the document does not opine):

1. Revenue concentration
2. Expense spike
3. Passthrough risk
4. Unallowable expenditure
5. Audit opinion
6. Going concern

Plus an `overall_verdict` (`COMPLIANT` / `REVIEW_REQUIRED` / `ESCALATE`), a list of flags with cited evidence, and a self-reported confidence. The prompt is defined in [src/llm/prompt_template.py](src/llm/prompt_template.py).

## 5. Corpus of source documents

The corpus consists of **12 publicly available audited financial statements and AUP reports** from real nonprofits. All documents are used solely for academic research; nothing in this repository asserts any compliance finding against the named organisations — the labels in the gold standard reflect what the documents *say*, not any judgement about the organisations themselves.

> **Note for the reader:** the PDFs are included in [data/pdfs/](data/pdfs/) for reproducibility. All were obtained from the organisations' public disclosures; the public source URL for each is listed below.

| # | File                                                     | Organisation / Report                | Public source |
|---|----------------------------------------------------------|--------------------------------------|-------------------------|
| 1 | `2024-CARE-USA-Financial-Statements_Final.pdf`           | CARE USA — FY2024 Financial Statements | https://www.care.org/financial-reports/ |
| 2 | `2024_Water.org_audited_financials.pdf`                  | Water.org — FY2024 Audited Financials  | https://water.org/about-us/financials/ |
| 3 | `8392233-FinancialStatement-1708722656073.pdf`           | River Network — Financial Statement  | https://www.rivernetwork.org/our-impact/annual-report/ |
| 4 | `BRAC-Liberia-Audited-Financial-Statements.pdf`          | BRAC Liberia — Audited Financials    | https://www.brac.net/stay-informed/annual-reports/ |
| 5 | `BRAC-Uganda-Audited-Financial-Statements.pdf`           | BRAC Uganda — Audited Financials     | https://www.brac.net/stay-informed/annual-reports/ |
| 6 | `FY23_Audit.pdf`                                         | Rocking the Boat, Inc. — FY23 Audit  | https://rockingtheboat.org/ |
| 7 | `FY24-25-ALC-Audit-Financial.pdf`                        | Asian Law Caucus — FY24-25 Audited Financials | https://www.asianlawcaucus.org/about/financial-annual-reports/financial-statements |
| 8 | `Justice-in-Aging-FY23-Audited-Financial-Statements.pdf` | Justice in Aging — FY23 Audited FS   | https://justiceinaging.org/go/annual-report-2023/index.html |
| 9 | `PATH-annual-report-2024.pdf`                            | PATH — Annual Report 2024            | https://www.path.org/who-we-are/finances/financial-documents/ |
| 10 | `Public-Citizen-Foundation-Inc.-FS-3.pdf`               | Public Citizen Foundation Inc. — FS  | https://www.citizen.org/about/annual-report/ |
| 11 | `Somos+2024+Audited+Financial+Statements+-+Final.pdf`   | Somos Mayfair — 2024 Audited Financials | https://www.somosmayfair.org/annualreports |
| 12 | `financial-statements-2024.pdf`                          | Save the Children Federation, Inc. — 2024 Financial Statements | https://www.savethechildren.org/us/about-us/ |

The gold-standard labels used for scoring are in [data/gold_standard/gold_standard.csv](data/gold_standard/gold_standard.csv).

## 6. Evaluation metrics

Every strategy is scored on four axes; the master table combines them.

| Axis           | Metric(s)                                                        | Source module                                                 |
|----------------|------------------------------------------------------------------|---------------------------------------------------------------|
| Quality        | Macro precision, recall, F1, accuracy vs gold standard           | [quality_scorer.py](src/evaluation/quality_scorer.py)         |
| Efficiency     | Average input tokens, cost per document, token reduction vs S1   | [efficiency_scorer.py](src/evaluation/efficiency_scorer.py)   |
| Consistency    | Cohen's κ across 3 independent runs of the same input            | [consistency_scorer.py](src/evaluation/consistency_scorer.py) |
| Faithfulness   | Share of cited evidence strings actually present in the input; hallucination rate | [faithfulness_scorer.py](src/evaluation/faithfulness_scorer.py) |

Each document is analysed **3 times per strategy** (`N_RUNS = 3`) at temperature 0.3 so that consistency can be measured.

## 7. Headline results

From [data/results/master_results_table.csv](data/results/master_results_table.csv):

| Strategy                  | Precision | Recall | F1    | Avg tokens | Cost / doc | Token reduction | Kappa | Faithfulness | Hallucination |
|---------------------------|-----------|--------|-------|------------|------------|-----------------|-------|--------------|---------------|
| S1 — Full text            | 0.830     | 0.810  | 0.808 | 14,101     | $0.0733    | 0%              | 0.939 | 1.00         | 0.0%          |
| S2 — Section filter       | 0.806     | 0.793  | 0.791 | 9,180      | $0.0487    | 35%             | 0.856 | 1.00         | 0.0%          |
| S5 — Extended extraction  | 0.705     | 0.690  | 0.684 | 4,957      | $0.0275    | 65%             | 0.932 | 1.00         | 0.0%          |
| S4 — Hybrid               | 0.759     | 0.707  | 0.691 | 1,336      | $0.0090    | 91%             | 0.898 | 1.00         | 0.0%          |
| S3 — Field extraction     | 0.716     | 0.672  | 0.655 | 1,124      | $0.0080    | 92%             | 0.898 | 1.00         | 0.0%          |

Full interpretation and discussion belong in the thesis document itself. In brief: S1 is the quality ceiling but is 8–9× more expensive than the aggressive extraction strategies; S2 retains most of S1's quality at ~⅔ of the cost; S3/S4 sacrifice ~15 F1 points for a ~90% cost reduction. All strategies were fully faithful (no hallucinated evidence strings) under the current faithfulness check.

## 8. Project structure

```
.
├── data/
│   ├── pdfs/                       # Source PDFs (not redistributed — see §5)
│   ├── gold_standard/              # Manually labelled ground truth
│   ├── ocr_cache/                  # Cached OCR output for scanned pages
│   └── results/                    # Raw run outputs (all_results.json) + master results table
├── src/
│   ├── ingestion/                  # PDF parsing, section classification, OCR fallback
│   ├── strategies/                 # The five input-preparation strategies (S1–S5)
│   ├── llm/                        # OpenAI client + prompt template
│   ├── output_processing/          # Verdict normalisation, faithfulness, hallucination detection
│   └── evaluation/                 # Quality / efficiency / consistency / faithfulness scorers
├── analysis/                       # Post-hoc analysis pipeline (see §10) — no API calls
│   ├── config.py                   # Shared constants: strategies, dimensions, paths, seed
│   ├── data_loader.py, id_mapping.py  # Load raw results/gold standard, reconcile document IDs
│   ├── validation.py                # Data-integrity checks (record/document/strategy counts, label validity)
│   ├── quality_analysis.py          # Precision/recall/F1/accuracy vs gold standard
│   ├── dimension_analysis.py        # Per-compliance-dimension breakdown
│   ├── document_analysis.py         # Per-document F1
│   ├── error_analysis.py            # False positive/negative case-level breakdown
│   ├── efficiency_analysis.py       # Token/cost analysis
│   ├── consistency_analysis.py      # Cohen's κ across runs
│   ├── faithfulness_analysis.py     # Evidence-citation faithfulness breakdown
│   ├── statistical_tests.py         # Token-vs-quality correlation, pairwise strategy changes
│   └── visualization.py             # Renders results/figures/ from the CSVs above
├── results/                         # Derived CSVs + figures produced by analysis/ (see §10)
├── parse.py, strategies.py, one_call.py, faithfulness.py, experiment.py  # Pipeline runners (see §9)
├── compute_metrics.py              # Re-scores existing results without re-calling the API
├── rescore_faithfulness.py         # Recomputes faithfulness on saved runs
├── requirements.txt
├── analysis/requirements-analysis.txt  # Extra deps for the analysis pipeline (numpy, pandas, scipy, scikit-learn, matplotlib)
└── README.md
```

## 9. Reproducing the experiment

### 9.1 Requirements

- Python 3.10+
- An OpenAI API key with access to `gpt-4o`
- The PDF corpus placed in `data/pdfs/` (see §5)

### 9.2 Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file at the project root:

```
OPENAI_API_KEY=sk-...
```

### 9.3 Run

The pipeline is split into five phases that map onto the thesis chapters:

| Script              | Purpose                                                       |
|---------------------|---------------------------------------------------------------|
| `parse.py`          | PDF parsing + section classification sanity check             |
| `strategies.py`     | Strategy preparation (no LLM calls)                            |
| `one_call.py`       | Single-run LLM sanity check                                    |
| `faithfulness.py`   | LLM call + hallucination/faithfulness detection                |
| `experiment.py`     | Full 3-run experiment on the entire corpus (produces `all_results.json`) |

```bash
python experiment.py          # Full experiment (calls the OpenAI API)
python compute_metrics.py     # Re-score saved results without any API calls
```

`compute_metrics.py` prints the four metric families and writes the master results table to `data/results/master_results_table.csv`.

## 10. Analysis pipeline

The scored, per-run output in `data/results/all_results.json` (produced by `experiment.py`, §9.3) is turned into the tables and figures used in the thesis by a second, **API-free** pipeline under [analysis/](analysis/). It only reads that committed JSON and the gold standard — reproducing the analysis requires no OpenAI calls, no cost, and no non-determinism.

### 10.1 Setup

```bash
pip install -r analysis/requirements-analysis.txt
```

### 10.2 Run order

The modules have a dependency chain: several downstream analyses read the evaluation dataset that `quality_analysis.py` writes to `results/quality/evaluation_dataset.csv`, and `statistical_tests.py`/`visualization.py` read the CSVs that the others produce. Run them in this order:

```bash
# 1. Validate the raw data before trusting any downstream number
python -m analysis.validation

# 2. Quality scoring — writes results/quality/, required by steps 3
python -m analysis.quality_analysis

# 3. Independent breakdowns (any order; each only needs step 2's output)
python -m analysis.dimension_analysis
python -m analysis.document_analysis
python -m analysis.error_analysis

# 4. Independent of quality_analysis — read data/results/all_results.json directly
python -m analysis.efficiency_analysis
python -m analysis.consistency_analysis
python -m analysis.faithfulness_analysis

# 5. Needs results/quality/ and results/efficiency/ from steps 2 and 4
python -m analysis.statistical_tests

# 6. Needs all of the above — renders results/figures/
python -m analysis.visualization
```

Each module is also independently runnable against the CSVs already committed under [results/](results/), so a reader can re-render figures or re-check a single metric family without rerunning the full chain. `analysis/config.py` holds the shared constants (strategy IDs, dimension names, expected record counts, the bootstrap random seed) that every module imports, so there is a single place to check or change them.

There is currently no single `run_all` entry point — `analysis/run_all.py` is an empty placeholder — so the six `python -m analysis.*` commands above must be run individually in the order shown.

## 11. Reproducibility notes and limitations

- **Model non-determinism.** Runs use temperature 0.3, not 0.0, in order to *measure* consistency rather than eliminate it. Results are therefore expected to vary slightly across executions; Cohen's κ across three runs is reported for exactly this reason.
- **Truncation.** S1 hits the input cap on the largest documents and is truncated with an explicit end-of-document marker. Truncation events are logged to `data/results/truncation_log.json` and discussed in the results chapter.
- **Faithfulness check.** The current check verifies that cited evidence strings appear (fuzzy match) in the input sent to the model. It does not verify that the *interpretation* of that evidence is correct — that remains a human judgement.
- **Corpus size.** 12 documents is a small sample by ML standards. This is a design choice: each document requires manual gold-standard labelling by the researcher, and the study is intended as an evaluation methodology rather than a production benchmark.
- **Deterministic analysis, non-deterministic generation.** The only randomness in the whole pipeline is the LLM call itself (§9.3, temperature 0.3). Everything downstream of `all_results.json` — every table and figure — is a pure, deterministic function of that committed file, the gold standard, and `analysis/config.py`'s fixed `RANDOM_SEED`; re-running the analysis pipeline (§10) on the committed data reproduces the thesis numbers exactly, without needing an API key.
- **Data-integrity checks.** `analysis/validation.py` asserts the expected record/document/strategy counts and label vocabulary before any metric is computed, so a corrupted or partial results file fails loudly instead of silently skewing a downstream table.
- **Document-ID reconciliation.** `analysis/id_mapping.py` and `analysis/config.py::EXPERIMENT_DIMENSION_MAP` reconcile two small naming inconsistencies between the raw experiment output and the gold-standard CSV (e.g. `passthrough_risk` vs. `pass_through_risk`); see the code comments there for the exact mapping if extending either schema.
- **Historical artifact.** `data/results/all_results_6docs_backup.json` is a snapshot from an earlier 6-document run kept for provenance; it is not read by any script and is not part of the reported results.
- **Human review required.** The verdicts produced here are a triage aid. Any real compliance action taken on a nonprofit grantee should go through the foundation's normal human review process.

## 12. Citation

If you refer to this work, please cite the thesis document:

```
Bandekar, C. (2026). An Empirical Evaluation of LLM Input Strategies for Token Efficiency, Evidential Traceability, and Compliance Reasoning in Nonprofit Grantee Financial Monitoring. Master's thesis.
```

