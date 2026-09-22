from pathlib import Path
import json
import re
import time

import pandas as pd
import pynvml
from llama_cpp import Llama


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

MODEL_PATH = SCRIPT_DIR / "Mistral-Nemo-Instruct-2407-Q4_K_S.gguf"
INPUT_FILE = PROJECT_ROOT / "Data" / "campagne.xlsx"

# Experimental output is stored in the current experiment directory.
OUTPUT_FILE = SCRIPT_DIR / "Classificazione_prompt.xlsx"

N_GPU_LAYERS = -1
CONTEXT_SIZE = 2048
MAX_TOKENS = 300
GPU_TEMPERATURE_LIMIT = 80
COOLDOWN_SECONDS = 60


# ============================================================
# GPU MONITORING
# ============================================================

pynvml.nvmlInit()


def get_gpu_temperature():
    """Return the temperature of the first NVIDIA GPU in Celsius."""
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetTemperature(
            handle,
            pynvml.NVML_TEMPERATURE_GPU,
        )
    except Exception:
        return 0


# ============================================================
# MODEL LOADING
# ============================================================

print("Loading model...")

llm = Llama(
    model_path=str(MODEL_PATH),
    n_gpu_layers=N_GPU_LAYERS,
    n_ctx=CONTEXT_SIZE,
    verbose=False,
)

if N_GPU_LAYERS != 0:
    print(f">>> STATUS: GPU enabled (n_gpu_layers = {N_GPU_LAYERS})")
else:
    print(">>> STATUS: GPU disabled (n_gpu_layers = 0)")


# ============================================================
# PROMPT
# ============================================================

# The prompt below is manually replaced according to the experimental run.
# The four prompt variants used in the experiments are available in
# "../prova_prompt".

def get_categorization_prompt(text):
    return f"""
    You are an expert analyst specialized in disinformation campaigns.
    
    Task:
    1. Read the entire text to understand its main narrative focus.
    2. Evaluate which category best represents the core intent or the most significant piece of information provided in the text.
    3. If the text provides multiple relevant details, assign the most central one as 'primary' and the secondary one as 'secondary'.
    
    Categories & Definitions:
    - ATTORI / SOGGETTI: Key entities, organizations, or groups behind the operation.
    - TATTICHE / METODO: Operational techniques, psychological manipulation, or methods used.
    - MEZZI / STRUMENTI: Platforms, digital tools, bots, or communication channels.
    - OBIETTIVO / TARGET: Intended victims, demographic groups, or strategic goals.
    - CONTENUTO / NARRATIVA: Specific content produced, themes, memes, or false narratives pushed.
    - IMPATTO / EFFETTO: Measurable consequences, social unrest, or behavioral changes.
    - VALIDAZIONE ANALITICA: How the campaign was debunked, verified, or analyzed by experts.
    - CONTESTO / ECOSISTEMA: Enabling environmental factors, political climate, or social conditions.
    - FINANZIAMENTO / MODELLO ECONOMICO: Financial structure, budgets, monetization, or resource allocation.

    Evidence to analyze: "{text}"

    Instructions (MUST FOLLOW):
    - Return ONLY a valid flat JSON object.
    - The JSON MUST contain ONLY two keys: "primary" and "secondary".
    - The values MUST be chosen strictly from the list of categories above.
    - DO NOT include any other keys, introductory text, markdown formatting, or explanations.
    - Example: {{"primary": "OBIETTIVO / TARGET", "secondary": "CONTENUTO / NARRATIVA"}}
    """


# ============================================================
# DATA LOADING
# ============================================================

print(f"Loading input file: {INPUT_FILE.name}")

df = pd.read_excel(INPUT_FILE)

evidence_columns = [
    column
    for column in df.columns
    if column not in ["NOME CAMPAGNA", "N° EVIDENZE"]
]


# ============================================================
# RESUME PREVIOUS EXECUTION
# ============================================================

if OUTPUT_FILE.exists():
    completed_df = pd.read_excel(OUTPUT_FILE)

    completed_evidence = set(
        zip(completed_df["CAMPAGNA"], completed_df["EVIDENZA"])
    )

    results = completed_df.to_dict("records")

    print(
        f"Resuming previous execution: "
        f"{len(completed_evidence)} evidence items already processed."
    )

else:
    completed_evidence = set()
    results = []

    print("Starting a new execution.")


# ============================================================
# MAIN CATEGORIZATION LOOP
# ============================================================

try:
    for _, row in df.iterrows():

        campaign_name = row["NOME CAMPAGNA"]

        for column in evidence_columns:

            evidence_text = row[column]

            if pd.isna(evidence_text) or str(evidence_text).strip() == "":
                continue

            evidence_text = str(evidence_text)

            # Skip evidence items already processed in a previous execution.
            if (campaign_name, evidence_text) in completed_evidence:
                continue

            # Monitor GPU temperature before inference.
            gpu_temperature = get_gpu_temperature()

            if gpu_temperature > GPU_TEMPERATURE_LIMIT:
                print(
                    f"GPU temperature is high ({gpu_temperature}°C). "
                    f"Cooling down for {COOLDOWN_SECONDS} seconds..."
                )
                time.sleep(COOLDOWN_SECONDS)

            try:
                response = llm.create_chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": get_categorization_prompt(evidence_text),
                        }
                    ],
                    # Temperature is manually changed according to the experimental run.
                    # Values used in the experiments: 0.0, 0.3, 0.6, 0.9.
                    temperature=0,
                    max_tokens=MAX_TOKENS,
                    stop=["<|endoftext|>", "</s>"],
                )

                raw_text = response["choices"][0]["message"]["content"]

                # Extract the JSON object from the model response.
                match = re.search(r"\{.*\}", raw_text, re.DOTALL)

                if match:
                    json_text = match.group(0)

                    try:
                        parsed_response = json.loads(json_text)

                    except json.JSONDecodeError:
                        # Attempt to repair a missing comma between JSON values.
                        repaired_text = json_text.replace('" "', '", "')

                        try:
                            parsed_response = json.loads(repaired_text)

                        except json.JSONDecodeError:
                            parsed_response = None

                    if parsed_response is not None:
                        primary = parsed_response.get("primary", "N/A")
                        secondary = parsed_response.get("secondary", "N/A")
                    else:
                        primary = "FORMAT_ERROR"
                        secondary = raw_text[:200]

                else:
                    primary = "FORMAT_ERROR"
                    secondary = raw_text[:200]

                results.append(
                    {
                        "CAMPAGNA": campaign_name,
                        "EVIDENZA": evidence_text,
                        "CATEGORIA_PRINCIPALE": primary,
                        "CATEGORIA_SECONDARIA": secondary,
                    }
                )

                # Save after each processed evidence item to allow safe resume.
                pd.DataFrame(results).to_excel(OUTPUT_FILE, index=False)

                print(f"Processed campaign: {campaign_name}")

            except Exception as error:
                results.append(
                    {
                        "CAMPAGNA": campaign_name,
                        "EVIDENZA": evidence_text,
                        "CATEGORIA_PRINCIPALE": "CRITICAL_ERROR",
                        "CATEGORIA_SECONDARIA": str(error),
                    }
                )

                pd.DataFrame(results).to_excel(OUTPUT_FILE, index=False)

                print(
                    f"Error while processing campaign "
                    f"'{campaign_name}': {error}"
                )

finally:
    pynvml.nvmlShutdown()


print("Categorization completed.")