import pandas as pd
import json
import os
import time
import pynvml
import re
from pathlib import Path
from llama_cpp import Llama


pynvml.nvmlInit()


def get_gpu_temp():
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetTemperature(
            handle,
            pynvml.NVML_TEMPERATURE_GPU
        )
    except:
        return 0


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

# Local Llama model.
MODEL_PATH = SCRIPT_DIR / "Llama-3-8B-Instruct-v0.10.Q5_K_M.gguf"

# Input dataset containing the campaign evidence.
FILE_EXCEL = PROJECT_ROOT / "Data" / "campagne.xlsx"

# Categorization results are stored in the current experiment directory.
OUTPUT_FILE = SCRIPT_DIR / "SingleLabel.xlsx"


# ---------------------------------------------------------------------
# MODEL INITIALIZATION
# ---------------------------------------------------------------------

print("Loading model on GPU...")

n_gpu_layers_config = -1

llm = Llama(
    model_path=str(MODEL_PATH),
    n_gpu_layers=-1,
    n_ctx=2048,
    verbose=False
)

if n_gpu_layers_config != 0:
    print(
        f">>> STATUS: GPU ENABLED "
        f"(n_gpu_layers = {n_gpu_layers_config})"
    )
else:
    print(">>> STATUS: GPU DISABLED (n_gpu_layers = 0)")


# ---------------------------------------------------------------------
# CATEGORIZATION PROMPT
# ---------------------------------------------------------------------

# The prompt below is manually replaced according to the experimental run.
# The four prompt variants used in the experiments are available in
# "Prova_Prompt".

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


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------

df = pd.read_excel(FILE_EXCEL)

colonne_evidenze = [
    c for c in df.columns
    if c not in ['NOME CAMPAGNA', 'N° EVIDENZE']
]


# ---------------------------------------------------------------------
# RESUME PREVIOUS EXECUTION
# ---------------------------------------------------------------------

if os.path.exists(OUTPUT_FILE):
    df_fatto = pd.read_excel(OUTPUT_FILE)

    gia_fatti = set(
        zip(df_fatto['CAMPAGNA'], df_fatto['EVIDENZA'])
    )

    results = df_fatto.to_dict('records')

    print(
        f"Resuming previous execution: "
        f"{len(gia_fatti)} evidence items already processed."
    )
else:
    gia_fatti = set()
    results = []
    print("Starting a new execution.")


# ---------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------

for index, row in df.iterrows():
    nome_campagna = row['NOME CAMPAGNA']

    for col in colonne_evidenze:

        testo = row[col]

        if pd.isna(testo) or str(testo).strip() == "":
            continue

        # Skip evidence that has already been processed.
        if (nome_campagna, testo) in gia_fatti:
            continue

        # Monitor GPU temperature before inference.
        temp = get_gpu_temp()

        if temp > 80:
            print(
                f"High GPU temperature ({temp}°C). "
                f"Pausing for 60 seconds..."
            )
            time.sleep(60)

        try:
            response = llm.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": get_categorization_prompt(testo)
                    }
                ],
                # Temperature is manually changed according to the
                # experimental run.
                # Values used in the experiments: 0.0, 0.3, 0.6, 0.9.
                temperature=0.9,
                max_tokens=200,
                seed=-1
            )

            raw_text = response['choices'][0]['message']['content']

            # Extract the first JSON object returned by the model.
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)

            if match:
                try:
                    data_raw = json.loads(match.group(0))

                except json.JSONDecodeError:
                    # Attempt a minimal repair when a comma is missing.
                    fixed_text = match.group(0).replace('" "', '", "')

                    try:
                        data_raw = json.loads(fixed_text)
                        primary = data_raw.get('primary', 'N/A')
                        secondary = data_raw.get('secondary', 'N/A')

                    except:
                        primary, secondary = (
                            "ERRORE_FORMATO",
                            raw_text[:200]
                        )

                else:
                    primary = data_raw.get('primary', 'N/A')
                    secondary = data_raw.get('secondary', 'N/A')

            else:
                primary, secondary = (
                    "ERRORE_FORMATO",
                    raw_text[:200]
                )

            results.append({
                'CAMPAGNA': nome_campagna,
                'EVIDENZA': testo,
                'CATEGORIA_PRINCIPALE': primary,
                'CATEGORIA_SECONDARIA': secondary
            })

            # Save after every processed evidence to preserve progress.
            pd.DataFrame(results).to_excel(
                OUTPUT_FILE,
                index=False
            )

            print(f"Saved: {nome_campagna}")

        except Exception as e:
            results.append({
                'CAMPAGNA': nome_campagna,
                'EVIDENZA': testo,
                'CATEGORIA_PRINCIPALE': "ERRORE_CRITICO",
                'CATEGORIA_SECONDARIA': str(e)
            })

            pd.DataFrame(results).to_excel(
                OUTPUT_FILE,
                index=False
            )

            print(f"Error processing {nome_campagna}: {e}")


pynvml.nvmlShutdown()

print("Process completed.")