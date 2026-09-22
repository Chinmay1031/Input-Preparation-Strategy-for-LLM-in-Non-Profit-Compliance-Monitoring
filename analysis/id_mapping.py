from analysis.data_loader import (
    load_experiment_results,
    load_gold_standard,
    summarize_experiment_records,
)


# Explicit, deterministic mapping between the document IDs
# used in the experiment results and the IDs used in the
# gold-standard annotations.
#
# IMPORTANT:
# The values on the right-hand side are copied from the
# document_id column of gold_standard.csv.
#
# Do not use fuzzy matching here. The mapping must be
# deterministic and auditable for the thesis analysis.

DOCUMENT_ID_MAP = {

    # CARE USA
    "2024-CARE-USA-Financial-Statements_Final":
        "CARE_USA_AuditedFinancials_2024",

    # Water.org
    "2024_Water.org_audited_financials":
        "WaterOrg_AuditedFinancials_2024",

    # BRAC Liberia
    "BRAC-Liberia-Audited-Financial-Statements":
        "BRAC_Liberia_AuditedFinancials_2023",

    # BRAC Uganda
    "BRAC-Uganda-Audited-Financial-Statements":
        "BRAC_Uganda_AuditedFinancials_2023",

    # River Network
    "8392233-FinancialStatement-1708722656073":
        "RiverNetwork_AuditedFinancials_2023",

    # Rocking the Boat
    "FY23_Audit":
        "RockingTheBoat_AuditedFinancials_2023",

    # Asian Law Caucus
    "FY24-25-ALC-Audit-Financial":
        "AsianLawCaucus_AuditedFinancials_2025",

    # Justice in Aging
    "Justice-in-Aging-FY23-Audited-Financial-Statements":
        "JusticeInAging_AuditedFinancials_2023",

    # PATH
    "PATH-annual-report-2024":
        "PATH_AnnualReport_2024",

    # Public Citizen Foundation
    "Public-Citizen-Foundation-Inc.-FS-3":
        "PublicCitizenFoundation_AuditedFinancials_2024",

    # SOMOS Mayfair
    "Somos+2024+Audited+Financial+Statements+-+Final":
        "SOMOSMayfair_AuditedFinancials_2024",

    # Save the Children Federation
    "financial-statements-2024":
        "SaveTheChildrenFederation_AuditedFinancials_2024",

    # 8392233 is River Network, as mapped above.
    # All 12 experiment document IDs are represented exactly once.
}


def get_experiment_document_ids(records):
    return sorted({
        record["doc_id"]
        for record in records
        if record.get("doc_id") is not None
    })


def get_gold_document_ids(gold_df):
    return sorted(
        gold_df["document_id"]
        .dropna()
        .astype(str)
        .unique()
    )


def validate_mapping(records, gold_df):
    """
    Validate the complete experiment-to-gold document mapping.

    The validation checks:

    1. Every experiment document has a mapping.
    2. Every mapped gold-standard ID exists.
    3. Every gold-standard document is used exactly once.
    4. The number of experiment and gold documents is equal.
    """

    experiment_ids = get_experiment_document_ids(records)
    gold_ids = get_gold_document_ids(gold_df)

    missing_experiment_mappings = [
        doc_id
        for doc_id in experiment_ids
        if doc_id not in DOCUMENT_ID_MAP
    ]

    if missing_experiment_mappings:
        raise ValueError(
            "The following experiment document IDs do not have "
            "a mapping:\n\n"
            + "\n".join(
                f"  {doc_id}"
                for doc_id in missing_experiment_mappings
            )
        )

    mapped_gold_ids = [
        DOCUMENT_ID_MAP[doc_id]
        for doc_id in experiment_ids
    ]

    missing_gold_ids = [
        gold_id
        for gold_id in mapped_gold_ids
        if gold_id not in gold_ids
    ]

    if missing_gold_ids:
        raise ValueError(
            "The following mapped gold-standard IDs do not "
            "exist in gold_standard.csv:\n\n"
            + "\n".join(
                f"  {gold_id}"
                for gold_id in missing_gold_ids
            )
        )

    duplicate_gold_ids = sorted({
        gold_id
        for gold_id in mapped_gold_ids
        if mapped_gold_ids.count(gold_id) > 1
    })

    if duplicate_gold_ids:
        raise ValueError(
            "Multiple experiment documents map to the same "
            "gold-standard document:\n\n"
            + "\n".join(
                f"  {gold_id}"
                for gold_id in duplicate_gold_ids
            )
        )

    unused_gold_ids = sorted(
        set(gold_ids) - set(mapped_gold_ids)
    )

    if unused_gold_ids:
        raise ValueError(
            "The following gold-standard documents are not "
            "represented by an experiment document:\n\n"
            + "\n".join(
                f"  {gold_id}"
                for gold_id in unused_gold_ids
            )
        )

    if len(experiment_ids) != len(gold_ids):
        raise ValueError(
            "The number of experiment documents and "
            "gold-standard documents differs.\n\n"
            f"Experiment documents: {len(experiment_ids)}\n"
            f"Gold documents:       {len(gold_ids)}"
        )

    return True


def apply_document_mapping(records):
    """
    Add gold_document_id to every experiment record.
    """

    mapped_records = []

    for record in records:

        record_copy = record.copy()

        experiment_id = record_copy.get("doc_id")

        if experiment_id not in DOCUMENT_ID_MAP:
            raise ValueError(
                f"No document mapping exists for: {experiment_id}"
            )

        record_copy["gold_document_id"] = DOCUMENT_ID_MAP[
            experiment_id
        ]

        mapped_records.append(record_copy)

    return mapped_records


def get_mapping_table(records):
    experiment_ids = get_experiment_document_ids(records)

    return [
        {
            "experiment_doc_id": experiment_id,
            "gold_document_id": DOCUMENT_ID_MAP[experiment_id],
        }
        for experiment_id in experiment_ids
    ]


if __name__ == "__main__":

    print("=" * 60)
    print("DOCUMENT ID MAPPING TEST")
    print("=" * 60)

    records = load_experiment_results()
    gold_df = load_gold_standard()

    summary = summarize_experiment_records(records)

    print("\nINPUT DATA")
    print("-" * 60)
    print(f"Experiment records:    {summary['records']}")
    print(f"Experiment documents:  {summary['documents']}")
    print(f"Gold-standard rows:    {len(gold_df)}")
    print(
        f"Gold-standard docs:    "
        f"{gold_df['document_id'].nunique()}"
    )

    print("\nVALIDATING MAPPING")
    print("-" * 60)

    validate_mapping(records, gold_df)

    print("✓ All experiment documents have a mapping.")
    print("✓ All mapped gold-standard IDs exist.")
    print("✓ No duplicate gold-standard mappings.")
    print("✓ Every gold-standard document is represented.")
    print("✓ Experiment and gold document counts match.")

    print("\nDOCUMENT MAPPING")
    print("-" * 60)

    mapping_table = get_mapping_table(records)

    for row in mapping_table:
        print(
            f"{row['experiment_doc_id']}"
            f"  -->  "
            f"{row['gold_document_id']}"
        )

    mapped_records = apply_document_mapping(records)

    print("\nMAPPED RECORDS")
    print("-" * 60)
    print(f"Original records: {len(records)}")
    print(f"Mapped records:   {len(mapped_records)}")

    print("\nEXPECTED STRUCTURE")
    print("-" * 60)
    print("12 documents")
    print("5 strategies")
    print("3 runs")
    print("12 × 5 × 3 = 180 records")

    print("\n" + "=" * 60)
    print("DOCUMENT ID MAPPING TEST PASSED")
    print("=" * 60)
