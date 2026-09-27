import re
import time
from pathlib import Path

import pandas as pd
import pynvml
from llama_cpp import Llama


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

INPUT_FILE = REPO_ROOT / "data" / "categorie_manuali_doppie.csv"
OUTPUT_FILE = SCRIPT_DIR / "Extracted_Tuples.xlsx"
MODEL_PATH = SCRIPT_DIR / "Llama-3-8B-Instruct-v0.10.Q5_K_M.gguf"


# ============================================================
# GPU MONITORING
# ============================================================

pynvml.nvmlInit()


def get_gpu_temp():
    """Returns the current GPU temperature."""
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetTemperature(
            handle,
            pynvml.NVML_TEMPERATURE_GPU
        )
    except:
        return 0


# ============================================================
# MODEL INFERENCE
# ============================================================

def model_run(llm, user_input):
    """Runs the model with the provided input."""

    stop_sequence = [
        "##END LIST##",
        "#END LIST#",
        "END LIST"
    ]

    output = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "Extract subject-verb-object tuples from text."
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        temperature=0.3,
        top_p=0.95,
        stop=stop_sequence,
        max_tokens=200
    )

    return output["choices"][0]["message"]["content"]


# ============================================================
# TUPLE EXTRACTION AND CLEANING
# ============================================================

def extract_all_tuples(text):
    """Extracts tuples following the Subject - Verb - Object format."""

    pattern = r'([^\n]+? - [^\n]+? - [^\n]+?)\n'

    return re.findall(pattern, text)


def clean_tuples(tuples):
    """Remove duplicate or malformed tuples."""

    cleaned_set = set()
    final_tuples = []

    for t in tuples:

        t = re.sub(
            r'\s+',
            ' ',
            t.strip()
        )

        if t.count(' - ') != 2:
            continue

        if any(
            word in t.lower()
            for word in [
                'is -',
                'are -',
                'was -',
                'it -',
                'there -'
            ]
        ):
            continue

        t_lower = t.lower()

        if t_lower not in cleaned_set:

            cleaned_set.add(t_lower)
            final_tuples.append(t)

    return final_tuples


# ============================================================
# PROMPT
# ============================================================

def create_full_prompt(text):
    """Creates the tuple extraction prompt."""

    prompt = f'''Instructions:
    1. Focus only on the main factual actions or events.
    2. Each tuple must contain:
    - A clear subject (the entity performing the action),
    - A precise verb (the action or relation),
    - A meaningful object (the entity affected or involved).
    3. Include factual statements even if they appear within direct quotes, **as long as the speaker is identifiable or can be inferred from context**.
    - In such cases, use the speaker as the subject of the tuple.
    - For example: if the sentence is 'John said, "We support the initiative."', extract: John - said - we support the initiative.
    4. Omit speculative, hypothetical, or unclear relationships.
    5. Use only the following format, one per line:
    Subject - Verb - Object
    6. Do not include explanations or paraphrasing.
    7. Write '##END LIST##' after the last tuple.

    Example:
    Text: 'Senator Smith stated, "We must protect the environment."'
    Output:
    Senator Smith - stated - we must protect the environment

    Example:
    Text: 'John gave a book to Mary.'
    Output:
    John - gave - a book to Mary
    ##END LIST##

    Now, extract tuples from this text: {text}'''

    return prompt


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # 1. MODEL AND GPU CONFIGURATION
    # ========================================================

    print("Loading model on GPU...")

    n_gpu_layers_config = -1

    llm = Llama(
        model_path=str(MODEL_PATH),
        n_gpu_layers=n_gpu_layers_config,
        n_ctx=2048,
        verbose=False
    )

    if n_gpu_layers_config != 0:

        print(
            f">>> STATUS: GPU ENABLED "
            f"(n_gpu_layers = {n_gpu_layers_config})"
        )

    else:

        print(
            ">>> STATUS: GPU DISABLED "
            "(n_gpu_layers = 0)"
        )


    # ========================================================
    # 2. DATA LOADING
    # ========================================================

    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        pynvml.nvmlShutdown()
        return

    df = pd.read_csv(
        INPUT_FILE,
        sep=";",
        encoding="utf-8-sig",
        quotechar='"',
        engine="python"
    )

    df.columns = df.columns.str.strip()

    print("\nInput columns:")
    print(df.columns.tolist())


    # ========================================================
    # 3. COLUMN VALIDATION
    # ========================================================

    required_columns = [
        "CAMPAGNA",
        "EVIDENZA",
        "CATEGORIA_MANUALE_1",
        "CATEGORIA_MANUALE_2"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print(
            "\nERROR: Missing required columns:"
        )

        for column in missing_columns:
            print(f"- {column}")

        print(
            "\nAvailable columns:"
        )

        print(df.columns.tolist())

        pynvml.nvmlShutdown()
        return


    # ========================================================
    # 4. RESUME OR START
    # ========================================================

    if OUTPUT_FILE.exists():
        df_done = pd.read_excel(OUTPUT_FILE)
        results = df_done.to_dict("records")

        already_processed = set(
            zip(df_done["CAMPAGNA"], df_done["EVIDENZA"])
        )

        print(
            f"\nResuming previous run: "
            f"{len(already_processed)} evidence items already processed."
        )
    else:
        already_processed = set()
        results = []
        print("\nStarting from scratch.")

    # ========================================================
    # 5. MAIN PROCESSING LOOP
    # ========================================================

    total_evidence = len(df)

    for index, row in df.iterrows():

        campaign_name = row["CAMPAGNA"]
        evidence_text = row["EVIDENZA"]

        primary_category = row["CATEGORIA_MANUALE_1"]
        secondary_category = row["CATEGORIA_MANUALE_2"]


        # Skip evidence already processed.
        if (campaign_name, evidence_text) in already_processed:
            continue


        # Skip empty evidence.
        if pd.isna(evidence_text) or str(evidence_text).strip() == "":
            continue


        # Citation cleaning:
        # - Removes content enclosed in round (...) or
        # - square [...] brackets before tuple extraction.

        cleaned_evidence = re.sub(
            r'\s*[\(\[].*?[\)\]]',
            '',
            str(evidence_text)
        )

        # GPU temperature monitoring
        temperature = get_gpu_temp()

        if temperature > 80:

            print(
                f"!!! High GPU temperature "
                f"({temperature}°C). "
                f"Pausing for 60 seconds..."
            )

            time.sleep(60)

        try:

            user_input = create_full_prompt(cleaned_evidence)
            response = model_run(llm, user_input)

            tuple_list = extract_all_tuples(response + "\n")
            cleaned_tuples = clean_tuples(tuple_list)

            for tuple_text in cleaned_tuples:

                results.append({
                    "CAMPAGNA": campaign_name,
                    "EVIDENZA": evidence_text,
                    "CATEGORIA_PRIMARIA": primary_category,
                    "CATEGORIA_SECONDARIA": secondary_category,
                    "TUPLA": tuple_text
                })


            # Save progress after each processed evidence item.
            pd.DataFrame(results).to_excel(OUTPUT_FILE, index=False)
            already_processed.add((campaign_name, evidence_text))

            print(
                f"Processed: {campaign_name} | "
                f"Evidence {index + 1}/{total_evidence} | "
                f"Tuples extracted: {len(cleaned_tuples)}"
            )


        except Exception as e:

            print(
                f"ERROR: {campaign_name} | "
                f"Evidence {index + 1}/{total_evidence} | "
                f"{e}"
            )


    # ========================================================
    # 6. COMPLETION
    # ========================================================

    pynvml.nvmlShutdown()

    print("\nTuple extraction completed!")
    print(f"Output saved to: {OUTPUT_FILE}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()