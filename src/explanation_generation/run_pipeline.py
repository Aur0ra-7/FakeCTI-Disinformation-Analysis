from pathlib import Path
import time
import pandas as pd
import pynvml

from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

from pipeline import (
    get_fake_by_id,
    build_tfidf_index,
    build_keyword_index,
    analyze_fake,
    generate_explanation
)


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

FAKE_PATH = REPO_ROOT / "data" / "FakeCTI.csv"
VALIDATIONS_90_PATH = (
    REPO_ROOT / "data" / "coverage" / "Analytical_Validations_90.xlsx"
)
EVIDENCES_319_PATH = (
    REPO_ROOT / "data" / "coverage" / "CTI_Evidences_319.xlsx"
)
SAMPLE_PATH = SCRIPT_DIR / "sample_1000.xlsx"
OUTPUT_PATH = SCRIPT_DIR / "pipeline_results_1000.xlsx"
MODEL_PATH = SCRIPT_DIR / "Mistral-Nemo-Instruct-2407-Q4_K_S.gguf"


# ============================================================
# INPUT VALIDATION
# ============================================================

required_files = [
    FAKE_PATH,
    VALIDATIONS_90_PATH,
    EVIDENCES_319_PATH,
    SAMPLE_PATH,
    MODEL_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

print("Loading FakeCTI...")

fake_df = pd.read_csv(
    FAKE_PATH,
    sep=";",
    encoding="latin1"
)


print("Loading 90 analytical validations...")

validations_90_df = pd.read_excel(
    VALIDATIONS_90_PATH
)


print("Loading 319 CTI evidences...")

evidences_319_df = pd.read_excel(
    EVIDENCES_319_PATH
)

print("Loading experimental sample...")

sample_df = pd.read_excel(
    SAMPLE_PATH,
    sheet_name="SAMPLE"
)


# ============================================================
# SAMPLE VALIDATION
# ============================================================

required_columns = [
    "SAMPLE_INDEX",
    "ID_FAKE",
    "CAMPAIGN",
    "TITLE",
    "TEXT"
]

missing_columns = [
    col
    for col in required_columns
    if col not in sample_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in sample_1000.xlsx: "
        f"{missing_columns}"
    )


if sample_df["ID_FAKE"].duplicated().any():
    raise ValueError(
        "Duplicate ID_FAKE values found in the sample."
    )


print(
    f"\nFake news items in sample: "
    f"{len(sample_df)}"
)

print(
    f"Campaigns represented: "
    f"{sample_df['CAMPAIGN'].nunique()}"
)


# ============================================================
# BUILD RETRIEVAL INDEXES
# ============================================================

print("\n" + "=" * 70)
print("BUILDING RETRIEVAL INDEXES")
print("=" * 70)


print(
    "Building TF-IDF index "
    "over the 90 analytical validations..."
)

tfidf_vectorizer, tfidf_matrix = (
    build_tfidf_index(
        validations_90_df
    )
)


print(
    "Building keyword index "
    "over the 319 CTI evidences..."
)

keyword_corpus = build_keyword_index(
    evidences_319_df
)


print("Retrieval indexes ready.")


# ============================================================
# LOAD E5
# ============================================================

print("\n" + "=" * 70)
print("LOADING E5")
print("=" * 70)

e5_model = SentenceTransformer(
    "intfloat/multilingual-e5-large",
    device="cpu"
)

print("E5 loaded.")


# ============================================================
# LOAD MISTRAL
# ============================================================

print("\n" + "=" * 70)
print("LOADING MISTRAL")
print("=" * 70)

llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=4096,
    n_gpu_layers=-1,
    verbose=False
)

print("Mistral loaded.")


# ============================================================
# GPU SAFETY
# ============================================================

GPU_MAX_TEMP = 80
GPU_RESUME_TEMP = 70
GPU_COOLDOWN_SECONDS = 60

pynvml.nvmlInit()

gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)


def check_gpu_temperature():

    while True:

        temperature = pynvml.nvmlDeviceGetTemperature(
            gpu_handle,
            pynvml.NVML_TEMPERATURE_GPU
        )

        print(f"GPU temperature: {temperature}°C")

        # Continue normally if the GPU temperature is safe
        if temperature < GPU_MAX_TEMP:
            return

        # Pause automatically if the GPU is too hot
        print(
            f"\nGPU temperature: {temperature}°C "
            f"(threshold: {GPU_MAX_TEMP}°C)."
        )

        print(
            f"Waiting {GPU_COOLDOWN_SECONDS} seconds "
            f"before checking again..."
        )

        time.sleep(GPU_COOLDOWN_SECONDS)

        temperature = pynvml.nvmlDeviceGetTemperature(
            gpu_handle,
            pynvml.NVML_TEMPERATURE_GPU
        )

        print(
            f"Temperature after cooldown: "
            f"{temperature}°C"
        )

        if temperature <= GPU_RESUME_TEMP:

            print(
                f"GPU cooled down to {temperature}°C "
                f"-> resuming batch execution."
            )

            return

        print(
            f"GPU is still above the resume "
            f"temperature ({GPU_RESUME_TEMP}°C). "
            f"Waiting again..."
        )


# ============================================================
# GROUNDING CONTEXT FOR THE JUDGE
# ============================================================

def get_grounding_context(
    stage,
    selected_ids,
    pattern,
    validations_90_df,
    evidences_319_df
):
    """
    Reconstruct the grounding context used to generate
    the final explanation.

    Stage 1 -> selected analytical validations
    Stage 2 -> selected CTI evidences
    Stage 3 -> campaign pattern
    """

    # For Stage 3, the grounding context is the pattern itself
    if stage == 3:
        return pattern if pattern else ""

    if not selected_ids:
        return ""

    # Select the appropriate evidence corpus
    if stage == 1:
        source_df = validations_90_df

    elif stage == 2:
        source_df = evidences_319_df

    else:
        return ""

    blocks = []

    for evidence_id in selected_ids:

        evidence_id = str(
            evidence_id
        ).strip()

        matches = source_df[
            source_df["ID"]
            .astype(str)
            .str.strip()
            == evidence_id
        ]

        if matches.empty:

            blocks.append(
                f"ID: {evidence_id}\n"
                f"[EVIDENCE NOT FOUND]"
            )

            continue

        row = matches.iloc[0]

        evidence_text = str(
            row["EVIDENZA"]
        )

        blocks.append(
            f"ID: {evidence_id}\n"
            f"Evidence: {evidence_text}"
        )

    return "\n\n---\n\n".join(
        blocks
    )


# ============================================================
# RESUME / CHECKPOINT
# ============================================================

if OUTPUT_PATH.exists():

    print("\n" + "=" * 70)
    print("CHECKPOINT FOUND")
    print("=" * 70)

    results_df = pd.read_excel(
        OUTPUT_PATH
    )

    completed_ids = set(
        results_df.loc[
            results_df["STATUS"] == "OK",
            "ID_FAKE"
        ]
        .astype(str)
        .str.strip()
    )

    print(
        f"Completed fake news items: "
        f"{len(completed_ids)}"
    )

    results = results_df.to_dict(
        orient="records"
    )

else:

    print("\nNo previous checkpoint found.")

    completed_ids = set()
    results = []


# ============================================================
# BATCH PIPELINE
# ============================================================

total = len(sample_df)

for position, row in sample_df.iterrows():

    sample_index = int(
        row["SAMPLE_INDEX"]
    )

    fake_id = str(
        row["ID_FAKE"]
    ).strip()


    # --------------------------------------------------------
    # SKIP ALREADY COMPLETED ITEMS
    # --------------------------------------------------------

    if fake_id in completed_ids:

        print(
            f"\n[{sample_index}/{total}] "
            f"{fake_id} already completed -> SKIP"
        )

        continue


    print("\n" + "=" * 70)

    print(
        f"FAKE {sample_index}/{total} - "
        f"{fake_id}"
    )

    print("=" * 70)


    try:

        # ====================================================
        # RETRIEVE FAKE FROM THE ORIGINAL DATASET
        # ====================================================

        fake = get_fake_by_id(
            fake_df,
            fake_id
        )


        print(
            f"Campaign: "
            f"{fake['campaign']}"
        )


        check_gpu_temperature()

        # ====================================================
        # THREE-STAGE ANALYSIS
        # ====================================================

        result = analyze_fake(

            fake=fake,

            fake_df=fake_df,

            validations_90_df=validations_90_df,

            tfidf_vectorizer=tfidf_vectorizer,

            tfidf_matrix=tfidf_matrix,

            keyword_corpus=keyword_corpus,

            e5_model=e5_model,

            llm=llm
        )


        # ====================================================
        # EXPLANATION GENERATION
        # ====================================================

        explanation = generate_explanation(

            fake=fake,

            result=result,

            llm=llm
        )


        # ====================================================
        # PREPARE RESULT DATA
        # ====================================================

        selected_ids = result.get(
            "selected_ids",
            []
        )

        if selected_ids is None:
            selected_ids = []


        if isinstance(
            selected_ids,
            list
        ):

            selected_ids_text = ", ".join(
                map(str, selected_ids)
            )

        else:

            selected_ids_text = str(
                selected_ids
            )


        pattern = result.get(
            "pattern",
            ""
        )

        if pattern is None:
            pattern = ""



        # ====================================================
        # GROUNDING CONTEXT FOR THE JUDGE
        # ====================================================

        grounding_context = get_grounding_context(
            stage=result["stage"],
            selected_ids=selected_ids,
            pattern=pattern,
            validations_90_df=validations_90_df,
            evidences_319_df=evidences_319_df
        )


        # ====================================================
        # SAVE RESULT
        # ====================================================

        record = {

            "SAMPLE_INDEX":
                sample_index,

            "ID_FAKE":
                fake_id,

            "CAMPAIGN":
                fake["campaign"],

            "FAKE_TEXT":
                fake["text"],

            "STAGE":
                result["stage"],

            "SOURCE_TYPE":
                result["source_type"],

            "SELECTED_IDS":
                selected_ids_text,

            "GROUNDING_CONTEXT":
                grounding_context,

            "PATTERN":
                pattern,

            "EXPLANATION":
                explanation if explanation else "",

            "STATUS":
                "OK",

            "ERROR":
                ""
        }

        # Replace any previous record for the same fake
        results = [
            r for r in results
            if str(r.get("ID_FAKE", "")).strip() != fake_id
        ]


        results.append(
            record
        )

        completed_ids.add(
            fake_id
        )


        print(
            f"\nStage: "
            f"{result['stage']}"
        )

        print(
            f"Source type: "
            f"{result['source_type']}"
        )

        if result["stage"] in [1, 2]:

            print(
                f"Evidence IDs: "
                f"{selected_ids_text}"
            )

        elif result["stage"] == 3:

            print(
                "\nPattern:"
            )

            print(pattern)


        print(
            "\nExplanation:"
        )

        print(explanation)


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        print(
            f"\nERROR FOR {fake_id}: "
            f"{e}"
        )


        record = {

            "SAMPLE_INDEX":
                sample_index,

            "ID_FAKE":
                fake_id,

            "CAMPAIGN":
                row["CAMPAIGN"],

            "FAKE_TEXT":
                row["TEXT"],

            "STAGE":
                "",

            "SOURCE_TYPE":
                "",

            "SELECTED_IDS":
                "",

            "GROUNDING_CONTEXT":
                "",

            "PATTERN":
                "",

            "EXPLANATION":
                "",

            "STATUS":
                "ERROR",

            "ERROR":
                str(e)
        }

        # Replace any previous record for the same fake
        results = [
            r for r in results
            if str(r.get("ID_FAKE", "")).strip() != fake_id
        ]

        results.append(
            record
        )


    # ========================================================
    # CHECKPOINT AFTER EACH FAKE
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "SAMPLE_INDEX"
    )

    results_df.to_excel(
        OUTPUT_PATH,
        index=False
    )


    print(
        f"\nCheckpoint saved -> "
        f"{OUTPUT_PATH}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("BATCH COMPLETED")
print("=" * 70)


results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "SAMPLE_INDEX"
)


# ============================================================
# FINAL STATISTICS
# ============================================================

ok_df = results_df[
    results_df["STATUS"] == "OK"
]

error_df = results_df[
    results_df["STATUS"] == "ERROR"
]


print(
    f"Successfully completed fake news items: "
    f"{len(ok_df)}"
)

print(
    f"Fake news items with errors: "
    f"{len(error_df)}"
)


if len(ok_df) > 0:

    print(
        "\nStage distribution:"
    )

    print(
        ok_df["STAGE"]
        .value_counts()
        .sort_index()
    )


print(
    f"\nResults saved to:\n"
    f"{OUTPUT_PATH}"
)