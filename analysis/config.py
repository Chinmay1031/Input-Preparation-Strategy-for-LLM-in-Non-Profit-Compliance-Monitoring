from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_RESULTS_DIR = PROJECT_ROOT / "data" / "results"

ALL_RESULTS_FILE = RAW_RESULTS_DIR / "all_results.json"
MASTER_RESULTS_FILE = RAW_RESULTS_DIR / "master_results_table.csv"
TRUNCATION_LOG_FILE = RAW_RESULTS_DIR / "truncation_log.json"

GOLD_STANDARD_DIR = PROJECT_ROOT / "data" / "gold_standard"

RESULTS_DIR = PROJECT_ROOT / "results"

VALIDATION_DIR = RESULTS_DIR / "validation"
QUALITY_DIR = RESULTS_DIR / "quality"
DIMENSION_DIR = RESULTS_DIR / "dimension"
DOCUMENT_DIR = RESULTS_DIR / "document"
EFFICIENCY_DIR = RESULTS_DIR / "efficiency"
CONSISTENCY_DIR = RESULTS_DIR / "consistency"
FAITHFULNESS_DIR = RESULTS_DIR / "faithfulness"
STATISTICS_DIR = RESULTS_DIR / "statistics"
ERROR_DIR = RESULTS_DIR / "errors"


STRATEGIES = [
    "S1_full",
    "S2_sections",
    "S3_fields",
    "S4_hybrid",
    "S5_extended",
]

STRATEGY_LABELS = {
    "S1_full": "S1 – Full text",
    "S2_sections": "S2 – Section filtering",
    "S3_fields": "S3 – Field extraction",
    "S4_hybrid": "S4 – Hybrid",
    "S5_extended": "S5 – Extended extraction",
}


# These names correspond to the gold-standard CSV columns.
DIMENSIONS = [
    "revenue_concentration",
    "expense_spike",
    "pass_through_risk",
    "unallowable_expenditure",
    "audit_opinion",
    "going_concern",
]

DIMENSION_LABELS = {
    "revenue_concentration": "Revenue concentration",
    "expense_spike": "Expense spike",
    "pass_through_risk": "Pass-through risk",
    "unallowable_expenditure": "Unallowable expenditure",
    "audit_opinion": "Audit opinion",
    "going_concern": "Going concern",
}


# The experiment results use "passthrough_risk", while the
# gold-standard CSV uses "pass_through_risk". All other
# dimension names are identical.

EXPERIMENT_DIMENSION_MAP = {
    "revenue_concentration": "revenue_concentration",
    "expense_spike": "expense_spike",
    "pass_through_risk": "passthrough_risk",
    "unallowable_expenditure": "unallowable_expenditure",
    "audit_opinion": "audit_opinion",
    "going_concern": "going_concern",
}


VALID_LABELS = {
    "CLEAR",
    "FLAG",
    "ESCALATE",
    "N/A",
}


POSITIVE_LABELS = {
    "FLAG",
    "ESCALATE",
}

NEGATIVE_LABELS = {
    "CLEAR",
}

NA_LABEL = "N/A"


EXPECTED_DOCUMENT_COUNT = 12
EXPECTED_STRATEGY_COUNT = 5
EXPECTED_RUN_COUNT = 3

EXPECTED_EXPERIMENT_RECORDS = (
    EXPECTED_DOCUMENT_COUNT
    * EXPECTED_STRATEGY_COUNT
    * EXPECTED_RUN_COUNT
)


INPUT_COST_PER_1000_TOKENS = 0.005
OUTPUT_COST_PER_1000_TOKENS = 0.015


CONFIDENCE_LEVEL = 0.95
ALPHA = 0.05

BOOTSTRAP_ITERATIONS = 10000

RANDOM_SEED = 42


def ensure_result_directories():
    directories = [
        RESULTS_DIR,
        VALIDATION_DIR,
        QUALITY_DIR,
        DIMENSION_DIR,
        DOCUMENT_DIR,
        EFFICIENCY_DIR,
        CONSISTENCY_DIR,
        FAITHFULNESS_DIR,
        STATISTICS_DIR,
        ERROR_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_strategy_label(strategy):
    return STRATEGY_LABELS.get(strategy, strategy)


def get_dimension_label(dimension):
    return DIMENSION_LABELS.get(dimension, dimension)
