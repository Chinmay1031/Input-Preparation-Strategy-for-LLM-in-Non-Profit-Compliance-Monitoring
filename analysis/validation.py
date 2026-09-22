from collections import Counter

from analysis.config import (
    DIMENSIONS,
    EXPERIMENT_DIMENSION_MAP,
    EXPECTED_DOCUMENT_COUNT,
    EXPECTED_EXPERIMENT_RECORDS,
    EXPECTED_RUN_COUNT,
    EXPECTED_STRATEGY_COUNT,
    STRATEGIES,
    VALID_LABELS,
    VALIDATION_DIR,
)
from analysis.data_loader import (
    load_experiment_results,
    load_gold_standard,
)
from analysis.id_mapping import (
    apply_document_mapping,
    validate_mapping,
)


def check_record_count(records):
    actual = len(records)

    if actual != EXPECTED_EXPERIMENT_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_EXPERIMENT_RECORDS} experiment "
            f"records, but found {actual}."
        )

    print(f"✓ Experiment record count: {actual}")


def check_documents(records):
    documents = {
        record.get("doc_id")
        for record in records
        if record.get("doc_id") is not None
    }

    actual = len(documents)

    if actual != EXPECTED_DOCUMENT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_DOCUMENT_COUNT} documents, "
            f"but found {actual}."
        )

    print(f"✓ Experiment document count: {actual}")


def check_strategies(records):
    strategies = {
        record.get("strategy")
        for record in records
        if record.get("strategy") is not None
    }

    expected = set(STRATEGIES)

    if strategies != expected:
        missing = sorted(expected - strategies)
        unexpected = sorted(strategies - expected)

        message = ["Strategy validation failed."]

        if missing:
            message.append(
                f"Missing strategies: {missing}"
            )

        if unexpected:
            message.append(
                f"Unexpected strategies: {unexpected}"
            )

        raise ValueError("\n".join(message))

    if len(strategies) != EXPECTED_STRATEGY_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_STRATEGY_COUNT} strategies, "
            f"but found {len(strategies)}."
        )

    print(
        f"✓ Strategy count and names: {len(strategies)}"
    )


def check_runs(records):
    runs = {
        record.get("run")
        for record in records
        if record.get("run") is not None
    }

    expected_runs = set(range(EXPECTED_RUN_COUNT))

    if runs != expected_runs:
        raise ValueError(
            f"Expected runs {sorted(expected_runs)}, "
            f"but found {sorted(runs)}."
        )

    print(f"✓ Run structure: {sorted(runs)}")


def check_document_strategy_run_completeness(records):
    counts = Counter(
        (
            record.get("doc_id"),
            record.get("strategy"),
            record.get("run"),
        )
        for record in records
    )

    duplicate_cells = [
        key
        for key, count in counts.items()
        if count != 1
    ]

    if duplicate_cells:
        raise ValueError(
            "Duplicate or missing experiment cells detected:\n"
            + "\n".join(
                f"  {key}: {counts[key]} records"
                for key in duplicate_cells
            )
        )

    documents = sorted({
        record["doc_id"]
        for record in records
    })

    missing_cells = []

    for document in documents:

        for strategy in STRATEGIES:

            for run in range(EXPECTED_RUN_COUNT):

                key = (
                    document,
                    strategy,
                    run,
                )

                if key not in counts:
                    missing_cells.append(key)

    if missing_cells:
        raise ValueError(
            "Missing experiment cells detected:\n"
            + "\n".join(
                f"  {key}"
                for key in missing_cells
            )
        )

    expected_cells = (
        EXPECTED_DOCUMENT_COUNT
        * EXPECTED_STRATEGY_COUNT
        * EXPECTED_RUN_COUNT
    )

    print(
        f"✓ Complete document × strategy × run grid: "
        f"{expected_cells} cells"
    )


def check_gold_standard(gold_df):
    expected_rows = EXPECTED_DOCUMENT_COUNT

    if len(gold_df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} gold-standard rows, "
            f"but found {len(gold_df)}."
        )

    unique_documents = gold_df["document_id"].nunique()

    if unique_documents != expected_rows:
        raise ValueError(
            "Gold-standard document IDs are not unique."
        )

    print(f"✓ Gold-standard rows: {len(gold_df)}")

    print(
        f"✓ Gold-standard document IDs: "
        f"{unique_documents}"
    )


def check_gold_standard_dimensions(gold_df):
    missing_columns = []

    for dimension in DIMENSIONS:

        label_column = f"{dimension}_label"
        evidence_column = f"{dimension}_evidence"

        if label_column not in gold_df.columns:
            missing_columns.append(label_column)

        if evidence_column not in gold_df.columns:
            missing_columns.append(evidence_column)

    if missing_columns:
        raise ValueError(
            "Missing gold-standard columns:\n"
            + "\n".join(
                f"  {column}"
                for column in missing_columns
            )
        )

    print(
        f"✓ All {len(DIMENSIONS)} compliance dimensions "
        f"contain label and evidence columns."
    )


def check_experiment_dimension_mapping(records):
    missing_dimensions = []

    for dimension in DIMENSIONS:

        experiment_dimension = EXPERIMENT_DIMENSION_MAP.get(
            dimension
        )

        if experiment_dimension is None:
            missing_dimensions.append(
                f"{dimension} -> <missing mapping>"
            )
            continue

        if not any(
            experiment_dimension in record
            for record in records
        ):
            missing_dimensions.append(
                f"{dimension} -> {experiment_dimension}"
            )

    if missing_dimensions:
        raise ValueError(
            "Experiment dimension mapping validation failed:\n"
            + "\n".join(
                f"  {item}"
                for item in missing_dimensions
            )
        )

    print(
        "✓ Experiment-to-gold dimension mapping is valid."
    )


def check_gold_labels(gold_df):
    invalid_values = {}

    for dimension in DIMENSIONS:

        column = f"{dimension}_label"

        values = (
            gold_df[column]
            .dropna()
            .astype(str)
            .str.strip()
        )

        invalid = sorted(
            set(values) - VALID_LABELS
        )

        if invalid:
            invalid_values[column] = invalid

    if invalid_values:

        lines = [
            "Invalid gold-standard labels detected:"
        ]

        for column, values in invalid_values.items():

            lines.append(
                f"  {column}: {values}"
            )

        raise ValueError("\n".join(lines))

    print(
        "✓ Gold-standard labels are valid "
        "(CLEAR / FLAG / ESCALATE / N/A)."
    )


def check_experiment_strategies(records):
    invalid = sorted({
        record.get("strategy")
        for record in records
        if record.get("strategy") not in STRATEGIES
    })

    if invalid:
        raise ValueError(
            f"Invalid experiment strategies found: {invalid}"
        )

    print(
        "✓ All experiment records use valid strategies."
    )


def check_experiment_runs(records):
    expected_runs = set(range(EXPECTED_RUN_COUNT))

    invalid = sorted({
        record.get("run")
        for record in records
        if record.get("run") not in expected_runs
    })

    if invalid:
        raise ValueError(
            f"Invalid experiment runs found: {invalid}"
        )

    print(
        "✓ All experiment records use valid run numbers."
    )


def validate_dataset():
    """
    Run the complete dataset validation.

    Returns
    -------
    records
        Experiment records with gold-standard document IDs.

    gold_df
        Gold-standard dataframe.
    """

    records = load_experiment_results()
    gold_df = load_gold_standard()

    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    print("\nEXPERIMENT STRUCTURE")
    print("-" * 60)

    check_record_count(records)
    check_documents(records)
    check_strategies(records)
    check_runs(records)
    check_experiment_strategies(records)
    check_experiment_runs(records)
    check_document_strategy_run_completeness(records)

    print("\nGOLD STANDARD")
    print("-" * 60)

    check_gold_standard(gold_df)
    check_gold_standard_dimensions(gold_df)
    check_gold_labels(gold_df)

    print("\nDIMENSION MAPPING")
    print("-" * 60)

    check_experiment_dimension_mapping(records)

    print("\nDOCUMENT MAPPING")
    print("-" * 60)

    validate_mapping(records, gold_df)

    print(
        "✓ Experiment-to-gold document mapping is complete."
    )

    mapped_records = apply_document_mapping(records)

    print("\nFINAL VALIDATION")
    print("-" * 60)

    if len(mapped_records) != EXPECTED_EXPERIMENT_RECORDS:
        raise ValueError(
            "Mapped record count changed unexpectedly."
        )

    print(
        f"✓ Mapped experiment records: "
        f"{len(mapped_records)}"
    )

    print("\n" + "=" * 60)
    print("DATASET VALIDATION PASSED")
    print("=" * 60)

    return mapped_records, gold_df


if __name__ == "__main__":
    validate_dataset()