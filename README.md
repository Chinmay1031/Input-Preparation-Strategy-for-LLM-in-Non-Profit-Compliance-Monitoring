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

> **Note for the reader:** the PDFs are included in [data/pdfs/](data/pdfs/) for reproducibility. All were obtained from the organisations' public disclosures. The table below is a placeholder — fill in the original public source URL for each document before publishing, so readers can verify provenance.

| # | File                                                     | Organisation / Report                | Public source (fill in) |
|---|----------------------------------------------------------|--------------------------------------|-------------------------|
| 1 | `2024-CARE-USA-Financial-Statements_Final.pdf`           | CARE USA — FY2024 Financial Statements | *[Add link]*          |
| 2 | `2024_Water.org_audited_financials.pdf`                  | Water.org — FY2024 Audited Financials  | *[Add link]*          |
| 3 | `8392233-FinancialStatement-1708722656073.pdf`           | River Network — Financial Statement  | *[Add link]*            |
| 4 | `BRAC-Liberia-Audited-Financial-Statements.pdf`          | BRAC Liberia — Audited Financials    | *[Add link]*            |
| 5 | `BRAC-Uganda-Audited-Financial-Statements.pdf`           | BRAC Uganda — Audited Financials     | *[Add link]*            |
| 6 | `FY23_Audit.pdf`                                         | Rocking the Boat, Inc. — FY23 Audit  | *[Add link]*            |
| 7 | `FY24-25-ALC-Audit-Financial.pdf`                        | ALC — FY24-25 Audited Financials     | *[Add link]*            |
| 8 | `Justice-in-Aging-FY23-Audited-Financial-Statements.pdf` | Justice in Aging — FY23 Audited FS   | *[Add link]*            |
| 9 | `PATH-annual-report-2024.pdf`                            | PATH — Annual Report 2024            | *[Add link]*            |
| 10 | `Public-Citizen-Foundation-Inc.-FS-3.pdf`               | Public Citizen Foundation Inc. — FS  | *[Add link]*            |
| 11 | `Somos+2024+Audited+Financial+Statements+-+Final.pdf`   | Somos — 2024 Audited Financials      | *[Add link]*            |
| 12 | `financial-statements-2024.pdf`                          | Save the Children Federation, Inc. — 2024 Financial Statements | *[Add link]*            |

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
│   └── results/                    # Raw run outputs + master results table
├── src/
│   ├── ingestion/                  # PDF parsing, section classification, OCR fallback
│   ├── strategies/                 # The five input-preparation strategies (S1–S5)
│   ├── llm/                        # OpenAI client + prompt template
│   ├── output_processing/          # Verdict normalisation, faithfulness, hallucination detection
│   └── evaluation/                 # Quality / efficiency / consistency / faithfulness scorers
├── test_phase1.py … test_phase5.py # Phased pipeline runners (see §9)
├── compute_metrics.py              # Re-scores existing results without re-calling the API
├── rescore_faithfulness.py         # Recomputes faithfulness on saved runs
├── requirements.txt
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

| Script            | Purpose                                                       |
|-------------------|---------------------------------------------------------------|
| `test_phase1.py`  | PDF parsing + section classification sanity check             |
| `test_phase2.py`  | Strategy preparation (no LLM calls)                            |
| `test_phase3.py`  | Single-run LLM sanity check                                    |
| `test_phase4.py`  | Full 3-run experiment on a subset                              |
| `test_phase5.py`  | Full 3-run experiment on the entire corpus (produces `all_results.json`) |

```bash
python test_phase5.py         # Full experiment (calls the OpenAI API)
python compute_metrics.py     # Re-score saved results without any API calls
```

`compute_metrics.py` prints the four metric families and writes the master results table to `data/results/master_results_table.csv`.

## 10. Reproducibility notes and limitations

- **Model non-determinism.** Runs use temperature 0.3, not 0.0, in order to *measure* consistency rather than eliminate it. Results are therefore expected to vary slightly across executions; Cohen's κ across three runs is reported for exactly this reason.
- **Truncation.** S1 hits the input cap on the largest documents and is truncated with an explicit end-of-document marker. Truncation events are logged to `data/results/truncation_log.json` and discussed in the results chapter.
- **Faithfulness check.** The current check verifies that cited evidence strings appear (fuzzy match) in the input sent to the model. It does not verify that the *interpretation* of that evidence is correct — that remains a human judgement.
- **Corpus size.** 12 documents is a small sample by ML standards. This is a design choice: each document requires manual gold-standard labelling by the researcher, and the study is intended as an evaluation methodology rather than a production benchmark.
- **Human review required.** The verdicts produced here are a triage aid. Any real compliance action taken on a nonprofit grantee should go through the foundation's normal human review process.

## 11. Citation

If you refer to this work, please cite the thesis document:

```
Bandekar, C. (2026). Input Strategies for Large Language Models in
Non-Profit Compliance Monitoring. Master's thesis. [Institution].
```

*[Add DOI / institutional repository link once the thesis is deposited.]*

## 12. License

*[Choose and add a license — e.g. MIT for the code, CC-BY-4.0 for the written thesis. The source PDFs remain the property of their respective publishers.]*
