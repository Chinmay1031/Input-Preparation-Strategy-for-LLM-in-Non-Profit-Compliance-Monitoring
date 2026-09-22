import json
from pathlib import Path

import pandas as pd

from analysis.config import (
    ALL_RESULTS_FILE,
    GOLD_STANDARD_DIR,
)


def load_experiment_results():
    """
    Load the raw experiment results from all_results.json.

    Returns
    -------
    list[dict]
        List of experiment records.
    """

    if not ALL_RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Experiment results file not found:\n{ALL_RESULTS_FILE}"
        )

    with open(ALL_RESULTS_FILE, "r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            "Expected all_results.json to contain a list of records."
        )

    return records


def find_gold_standard_file():
    """
    Find CSV files inside data/gold_standard/.

    The function does not assume a specific filename.
    Exactly one CSV is expected.
    """

    if not GOLD_STANDARD_DIR.exists():
        raise FileNotFoundError(
            f"Gold-standard directory not found:\n{GOLD_STANDARD_DIR}"
        )

    csv_files = sorted(GOLD_STANDARD_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in:\n{GOLD_STANDARD_DIR}"
        )

    if len(csv_files) > 1:
        files = "\n".join(str(path.name) for path in csv_files)

        raise RuntimeError(
            "Multiple CSV files were found in the gold-standard directory. "
            "Please specify which one should be used.\n\n"
            f"{files}"
        )

    return csv_files[0]


def load_gold_standard():
    """
    Load the gold-standard CSV.

    Returns
    -------
    pandas.DataFrame
        Gold-standard annotations.
    """

    gold_file = find_gold_standard_file()

    gold_df = pd.read_csv(gold_file)

    if gold_df.empty:
        raise ValueError(
            f"Gold-standard file is empty:\n{gold_file}"
        )

    return gold_df


def summarize_experiment_records(records):
    if not records:
        return {
            "records": 0,
            "documents": 0,
            "strategies": 0,
            "runs": 0,
        }

    documents = sorted({
        record.get("doc_id")
        for record in records
        if record.get("doc_id") is not None
    })

    strategies = sorted({
        record.get("strategy")
        for record in records
        if record.get("strategy") is not None
    })

    runs = sorted({
        record.get("run")
        for record in records
        if record.get("run") is not None
    })

    return {
        "records": len(records),
        "documents": len(documents),
        "strategies": len(strategies),
        "runs": len(runs),
    }


def summarize_gold_standard(gold_df):
    return {
        "rows": len(gold_df),
        "columns": len(gold_df.columns),
        "columns_list": list(gold_df.columns),
    }


if __name__ == "__main__":

    print("=" * 60)
    print("DATA LOADER TEST")
    print("=" * 60)

    records = load_experiment_results()

    experiment_summary = summarize_experiment_records(records)

    print("\nEXPERIMENT RESULTS")
    print("-" * 60)
    print(f"Records:    {experiment_summary['records']}")
    print(f"Documents:  {experiment_summary['documents']}")
    print(f"Strategies: {experiment_summary['strategies']}")
    print(f"Runs:       {experiment_summary['runs']}")

    print("\nDocument IDs:")
    for doc_id in sorted({
        record.get("doc_id")
        for record in records
        if record.get("doc_id") is not None
    }):
        print(f"  {doc_id}")

    gold_file = find_gold_standard_file()
    gold_df = load_gold_standard()

    gold_summary = summarize_gold_standard(gold_df)

    print("\nGOLD STANDARD")
    print("-" * 60)
    print(f"File:    {gold_file.name}")
    print(f"Rows:    {gold_summary['rows']}")
    print(f"Columns: {gold_summary['columns']}")

    print("\nColumns:")
    for column in gold_summary["columns_list"]:
        print(f"  {column}")

    print("\nGold-standard preview:")
    print(gold_df.head().to_string(index=False))

    print("\n" + "=" * 60)
    print("DATA LOADER TEST COMPLETE")
    print("=" * 60)
