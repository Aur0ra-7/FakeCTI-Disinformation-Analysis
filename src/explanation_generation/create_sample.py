from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

FILE_INPUT = REPO_ROOT / "data" / "FakeCTI.csv"
FILE_OUTPUT = SCRIPT_DIR / "sample_1000.xlsx"

# Reuse the official experimental sample included in the repository.
# Set REFERENCE_SAMPLE = None to generate a new sample from FakeCTI
# using the same sampling strategy.
REFERENCE_SAMPLE = SCRIPT_DIR / "sample_1000.xlsx"

N_TOTAL = 1000
MIN_PER_CAMPAIGN = 5
RANDOM_STATE = 42


# ============================================================
# HELPERS
# ============================================================

def normalize_id(value):
    """Normalize FakeCTI IDs for validation and comparison."""
    value = str(value).strip().upper()

    if value.startswith("F"):
        value = value[1:]

    return value


def validate_dataset(df):
    """Validate the columns and IDs required for sampling."""
    required_columns = [
        "ID",
        "CAMPAIGN",
        "TITLE",
        "TEXT",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in FakeCTI.csv: {missing_columns}"
        )

    normalized_ids = df["ID"].apply(normalize_id)

    if normalized_ids.duplicated().any():
        raise ValueError("Duplicate IDs found in FakeCTI.csv.")

    if N_TOTAL > len(df):
        raise ValueError(
            f"Requested sample size ({N_TOTAL}) exceeds "
            f"dataset size ({len(df)})."
        )


def validate_sample(sample_df, dataset_df):
    """Validate a sample against the original FakeCTI dataset."""
    required_columns = [
        "SAMPLE_INDEX",
        "ID_FAKE",
        "CAMPAIGN",
        "TITLE",
        "TEXT",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in sample_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in sample: {missing_columns}"
        )

    if len(sample_df) != N_TOTAL:
        raise ValueError(
            f"Expected {N_TOTAL} fake news items in the sample, "
            f"found {len(sample_df)}."
        )

    normalized_sample_ids = sample_df["ID_FAKE"].apply(normalize_id)

    if normalized_sample_ids.duplicated().any():
        raise ValueError(
            "Duplicate ID_FAKE values found in the sample."
        )

    dataset_ids = set(dataset_df["ID"].apply(normalize_id))
    sample_ids = set(normalized_sample_ids)
    missing_ids = sample_ids - dataset_ids

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} sample IDs were not found in FakeCTI.csv."
        )

    expected_indexes = list(range(1, N_TOTAL + 1))
    actual_indexes = sample_df["SAMPLE_INDEX"].tolist()

    if actual_indexes != expected_indexes:
        raise ValueError(
            "SAMPLE_INDEX must contain consecutive values from "
            f"1 to {N_TOTAL}."
        )


def compute_target_allocation(df):
    """Allocate the target sample across campaigns."""
    campaign_counts = df["CAMPAIGN"].value_counts()
    max_available = campaign_counts.copy()

    samples_per_campaign = pd.Series(
        index=campaign_counts.index,
        data=np.minimum(MIN_PER_CAMPAIGN, max_available),
        dtype=int,
    )

    remaining = N_TOTAL - int(samples_per_campaign.sum())

    if remaining < 0:
        raise ValueError(
            "Minimum campaign quota exceeds the requested sample size."
        )

    while remaining > 0:
        capacity = max_available - samples_per_campaign
        eligible = capacity[capacity > 0]

        if eligible.empty:
            raise ValueError(
                f"Unable to allocate {N_TOTAL} fake news items."
            )

        proportions = (
            campaign_counts[eligible.index]
            / campaign_counts[eligible.index].sum()
        )

        additional = proportions * remaining
        additional_floor = np.floor(additional).astype(int)
        additional_floor = np.minimum(additional_floor, eligible)

        if additional_floor.sum() == 0:
            remainders = additional - np.floor(additional)
            campaign = remainders.idxmax()
            samples_per_campaign[campaign] += 1
            remaining -= 1
        else:
            assigned = int(additional_floor.sum())
            samples_per_campaign.loc[
                additional_floor.index
            ] += additional_floor
            remaining -= assigned

    if int(samples_per_campaign.sum()) != N_TOTAL:
        raise RuntimeError(
            "Target allocation does not match the requested sample size."
        )

    return campaign_counts, samples_per_campaign


def generate_sample(df, samples_per_campaign):
    """Generate a stratified random sample using the target allocation."""
    sampled_campaigns = []

    for campaign, n_samples in samples_per_campaign.items():
        campaign_df = df[df["CAMPAIGN"] == campaign]

        if len(campaign_df) < n_samples:
            raise ValueError(
                f"Campaign '{campaign}' requires {n_samples} fake news "
                f"items, but only {len(campaign_df)} are available."
            )

        sampled = campaign_df.sample(
            n=int(n_samples),
            random_state=RANDOM_STATE,
        )

        sampled_campaigns.append(sampled)

    sample_df = pd.concat(
        sampled_campaigns,
        ignore_index=True,
    )

    sample_df = sample_df[
        ["ID", "CAMPAIGN", "TITLE", "TEXT"]
    ].copy()

    sample_df.rename(
        columns={"ID": "ID_FAKE"},
        inplace=True,
    )

    # Keep the F-prefixed ID representation used by the experiment.
    sample_df["ID_FAKE"] = (
        "F"
        + sample_df["ID_FAKE"]
        .astype(str)
        .str.strip()
        .str.removeprefix("F")
        .str.removeprefix("f")
    )

    sample_df.insert(
        0,
        "SAMPLE_INDEX",
        range(1, len(sample_df) + 1),
    )

    return sample_df


def build_distribution(dataset_df, sample_df):
    """Build dataset/sample campaign distribution statistics."""
    dataset_distribution = (
        dataset_df["CAMPAIGN"]
        .value_counts()
        .rename_axis("CAMPAIGN")
        .reset_index(name="N_DATASET")
    )

    sample_distribution = (
        sample_df["CAMPAIGN"]
        .value_counts()
        .rename_axis("CAMPAIGN")
        .reset_index(name="N_SAMPLE")
    )

    distribution_df = dataset_distribution.merge(
        sample_distribution,
        on="CAMPAIGN",
        how="left",
    )

    distribution_df["N_SAMPLE"] = (
        distribution_df["N_SAMPLE"]
        .fillna(0)
        .astype(int)
    )

    distribution_df["PCT_DATASET"] = (
        distribution_df["N_DATASET"] / len(dataset_df) * 100
    )

    distribution_df["PCT_SAMPLE"] = (
        distribution_df["N_SAMPLE"] / len(sample_df) * 100
    )

    distribution_df["REPRESENTATION_RATIO"] = (
        distribution_df["PCT_SAMPLE"]
        / distribution_df["PCT_DATASET"]
    )

    return distribution_df.sort_values(
        by="N_DATASET",
        ascending=False,
    ).reset_index(drop=True)


def build_allocation(campaign_counts, sample_df):
    """Build the final campaign allocation table."""
    final_distribution = (
        sample_df["CAMPAIGN"]
        .value_counts()
        .reindex(campaign_counts.index, fill_value=0)
    )

    return pd.DataFrame(
        {
            "CAMPAIGN": campaign_counts.index,
            "N_DATASET": campaign_counts.values,
            "N_SAMPLE": final_distribution.values,
        }
    )


def build_metadata(dataset_df, sample_df, mode):
    """Build metadata describing the sampling configuration."""
    return pd.DataFrame(
        {
            "PARAMETER": [
                "Dataset size",
                "Sample size",
                "Number of campaigns in dataset",
                "Number of campaigns in sample",
                "Minimum per campaign",
                "Random state",
                "Mode",
                "Sampling strategy",
            ],
            "VALUE": [
                len(dataset_df),
                len(sample_df),
                dataset_df["CAMPAIGN"].nunique(),
                sample_df["CAMPAIGN"].nunique(),
                MIN_PER_CAMPAIGN,
                RANDOM_STATE,
                mode,
                (
                    "Disproportionate stratified random sampling with "
                    "a minimum campaign quota followed by proportional "
                    "allocation of the remaining observations."
                ),
            ],
        }
    )


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

if not FILE_INPUT.exists():
    raise FileNotFoundError(
        f"Required file not found: {FILE_INPUT}"
    )

print("=" * 70)
print("LOADING FAKECTI")
print("=" * 70)

df = pd.read_csv(
    FILE_INPUT,
    sep=";",
    encoding="latin1",
)

validate_dataset(df)

print(f"Total fake news items in dataset: {len(df)}")
print(f"Total campaigns: {df['CAMPAIGN'].nunique()}")


# ============================================================
# TARGET ALLOCATION
# ============================================================

campaign_counts, samples_per_campaign = compute_target_allocation(df)

print("\n" + "=" * 70)
print("TARGET ALLOCATION")
print("=" * 70)

allocation_target = pd.DataFrame(
    {
        "N_DATASET": campaign_counts,
        "N_TARGET": samples_per_campaign,
    }
)

print(allocation_target)


# ============================================================
# LOAD REFERENCE SAMPLE OR GENERATE A NEW SAMPLE
# ============================================================

if REFERENCE_SAMPLE is not None:
    if not REFERENCE_SAMPLE.exists():
        raise FileNotFoundError(
            f"Reference sample not found: {REFERENCE_SAMPLE}\n"
            "Set REFERENCE_SAMPLE = None to generate a new sample."
        )

    print("\nUsing the official experimental sample.")

    sample_df = pd.read_excel(
        REFERENCE_SAMPLE,
        sheet_name="SAMPLE",
    )

    mode = "Official experimental sample"

else:
    print("\nGenerating a new sample from FakeCTI...")

    sample_df = generate_sample(
        df,
        samples_per_campaign,
    )

    mode = "New sample generated from FakeCTI"


# ============================================================
# FINAL VALIDATION
# ============================================================

validate_sample(sample_df, df)

print("\n" + "=" * 70)
print("FINAL CHECK")
print("=" * 70)
print(f"Final sample size: {len(sample_df)}")
print("No duplicate ID_FAKE values found.")
print("All sample IDs were found in FakeCTI.csv.")


# ============================================================
# OUTPUT TABLES
# ============================================================

distribution_df = build_distribution(
    df,
    sample_df,
)

allocation_df = build_allocation(
    campaign_counts,
    sample_df,
)

metadata_df = build_metadata(
    df,
    sample_df,
    mode,
)


# ============================================================
# SAVE
# ============================================================

# The official sample is already stored in sample_1000.xlsx.
# In reference mode, the file is only validated and reused so that
# the exact experimental sample remains unchanged.
if REFERENCE_SAMPLE is None:
    with pd.ExcelWriter(
        FILE_OUTPUT,
        engine="openpyxl",
    ) as writer:
        sample_df.to_excel(
            writer,
            sheet_name="SAMPLE",
            index=False,
        )
        distribution_df.to_excel(
            writer,
            sheet_name="DISTRIBUTION",
            index=False,
        )
        allocation_df.to_excel(
            writer,
            sheet_name="ALLOCATION",
            index=False,
        )
        metadata_df.to_excel(
            writer,
            sheet_name="METADATA",
            index=False,
        )

    print("\n" + "=" * 70)
    print("COMPLETED")
    print("=" * 70)
    print(f"New sample saved to:\n{FILE_OUTPUT}")

else:
    print("\n" + "=" * 70)
    print("COMPLETED")
    print("=" * 70)
    print(f"Official sample validated:\n{REFERENCE_SAMPLE}")
    print(
        "Set REFERENCE_SAMPLE = None to generate and save "
        "a new sample using the same sampling strategy."
    )
