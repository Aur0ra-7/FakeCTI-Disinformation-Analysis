from pathlib import Path
import re
import time

import pandas as pd
import pynvml
from llama_cpp import Llama


# ============================================================
# GPU INITIALIZATION
# ============================================================
pynvml.nvmlInit()


def get_gpu_temp():
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        )
    except:
        return 0


# ============================================================
# FILES AND PARAMETERS
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "coverage"

MODEL_PATH = SCRIPT_DIR / "Mistral-Nemo-Instruct-2407-Q4_K_S.gguf"
FILE_VALIDAZIONI = DATA_DIR / "Analytical_Validations_90.xlsx"
FILE_FAKE = DATA_DIR / "Fake101_Benchmark.xlsx"
OUTPUT_FILE = SCRIPT_DIR / "Keyword_Fake101.xlsx"

TOP_K_RETRIEVAL = 10
MAX_CONTEXT = 4096
SAFETY_MARGIN = 200


# ============================================================
# MODEL LOADING
# ============================================================
print("Loading Mistral model on GPU...")
llm = Llama(
    model_path=str(MODEL_PATH),
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=False,
)
print(">>> LLM LOADED")


# ============================================================
# VALIDATION CORPUS LOADING
# ============================================================
df_val = pd.read_excel(FILE_VALIDAZIONI)
df_val.columns = df_val.columns.str.strip()

stop_words_set = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "with", "by", "of", "is", "are", "was", "were", "this",
    "that", "it", "as", "from"
}

validazioni_corpus = []

for _, row in df_val.iterrows():
    v_id = str(row["ID"]).strip()
    v_campagna = str(row["CAMPAGNA"]).strip()
    v_testo = str(row["EVIDENZA"]).strip()

    parole = re.findall(r"\b\w+\b", v_testo.lower())
    keywords = {
        p
        for p in parole
        if len(p) > 3 and p not in stop_words_set
    }

    validazioni_corpus.append(
        {
            "id": v_id,
            "campagna": v_campagna,
            "testo": v_testo,
            "keywords": keywords,
        }
    )

print(f">>> Validation corpus loaded: {len(validazioni_corpus)} elements")


# ============================================================
# KEYWORD RETRIEVAL
# ============================================================
def keyword_retrieve(fake_text, top_k=TOP_K_RETRIEVAL):
    fake_words = re.findall(r"\b\w+\b", fake_text.lower())

    fake_keywords = {
        p
        for p in fake_words
        if len(p) > 3 and p not in stop_words_set
    }

    risultati = []

    for val in validazioni_corpus:
        comuni = fake_keywords.intersection(val["keywords"])
        kw_score = len(comuni)

        risultati.append(
            {
                **val,
                "total_score": kw_score,
                "num_comuni": len(comuni),
            }
        )

    risultati.sort(
        key=lambda x: x["total_score"],
        reverse=True,
    )

    return risultati[:top_k]


# ============================================================
# PROMPT
# ============================================================
def create_prompt(fake_id, campaign, fake_text_placeholder, validations):
    validation_text = ""

    for v in validations:
        validation_text += (
            f"\nID: {v['id']}\n"
            f"Campaign: {v['campagna']}\n"
            f"Validation: {v['testo']}\n"
            f"---\n"
        )

    return f"""
You are an expert Cyber Threat Intelligence analyst.
Your task is to determine whether any candidate validation refutes the specific claim made in the FAKE NEWS.

FAKE NEWS:
ID: {fake_id}
Campaign: {campaign}
Text: {fake_text_placeholder}

CANDIDATE VALIDATIONS:
{validation_text}

INSTRUCTIONS:
Select a validation ID ONLY if it meets ONE of these two strict criteria:

A. DIRECT REFUTATION
   The validation provides factual evidence about the SAME specific claim 
   (same event, same actors, same location, same evidence) and shows it is false.

B. DIRECT-PATTERN REFUTATION  
   The validation demonstrates that the EXACT METHODOLOGY or TYPE OF EVIDENCE 
   used in the fake news (e.g., "giant skeleton discovered at archaeological site") 
   has been previously and definitively exposed as a known fabrication pattern, 
   AND this exposure explicitly undermines the credibility of the current claim.
   The validation must name the specific pattern or precedent; broad statements 
   about the topic are insufficient.

REJECTION RULES:
- Do NOT select for general topic similarity, shared keywords, or same campaign.
- Do NOT select if the validation merely discusses a related historical case 
  without establishing that the current claim follows the SAME debunked pattern.
- Do NOT infer connections not explicitly stated in the validation text.
- If uncertain, output NONE. Never force a match.

OUTPUT FORMAT:
Return ONLY the matching IDs separated by commas (e.g., V06, V07) or NONE.
"""


# ============================================================
# SAMPLE PROCESSING
# ============================================================
df_fake_test = pd.read_excel(FILE_FAKE)
risultati = []

for index, row in df_fake_test.reset_index(drop=True).iterrows():
    print("\n" + "=" * 70)
    print(f"FAKE {index + 1}/{len(df_fake_test)}")
    print("=" * 70)

    fake_id = str(row.get("ID_FAKE", f"F{index + 1}")).strip()
    campaign = str(row.get("CAMPAIGN", "")).strip()
    fake_text = str(row.get("TEXT", "")).strip()

    print(f"ID: {fake_id}")
    print(f"Campaign: {campaign}")
    print(f"Fake text: {fake_text[:300]}...")

    # GPU temperature monitoring
    temp = get_gpu_temp()
    print(f"GPU temperature: {temp}°C")

    if temp > 80:
        print("!!! High GPU temperature. Pausing for 60 seconds...")
        time.sleep(60)

    # Retrieve the Top-K candidate validations
    print(
        f"\nKeyword retrieval "
        f"(TOP {TOP_K_RETRIEVAL} candidates)..."
    )

    top_validations = keyword_retrieve(fake_text)

    # Dynamic token calculation and safe truncation
    prompt_base = create_prompt(
        fake_id,
        campaign,
        "[PLACEHOLDER]",
        top_validations,
    )

    base_tokens = len(
        llm.tokenize(prompt_base.encode("utf-8"))
    )

    max_allowed_fake = (
        MAX_CONTEXT
        - base_tokens
        - SAFETY_MARGIN
    )

    fake_tokens_full = llm.tokenize(
        fake_text.encode("utf-8")
    )

    if len(fake_tokens_full) > max_allowed_fake:
        print(
            "Fake text too long"
            f" ({len(fake_tokens_full)} tokens). Truncating to"
            f" {max_allowed_fake} tokens..."
        )

        final_fake_text = llm.detokenize(
            fake_tokens_full[:max_allowed_fake]
        ).decode(
            "utf-8",
            errors="ignore",
        )
    else:
        final_fake_text = fake_text

    # Build the final prompt
    prompt = create_prompt(
        fake_id,
        campaign,
        final_fake_text,
        top_validations,
    )

    token_prompt = len(
        llm.tokenize(prompt.encode("utf-8"))
    )

    print(
        f"Total prompt tokens: "
        f"{token_prompt}/{MAX_CONTEXT}"
    )

    # Mistral inference
    try:
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
            max_tokens=150,
        )

        raw_output = (
            response["choices"][0]["message"]["content"].strip()
        )

    except Exception as e:
        print(f"MODEL ERROR: {e}")
        raw_output = "NONE"

    print(f"\n>>> MODEL RESPONSE: {raw_output}")

    # Extract and validate IDs returned by the model
    trovati = re.findall(
        r"\bV\d+\b",
        raw_output.upper(),
    )

    trovati = list(dict.fromkeys(trovati))

    id_validi = {
        str(v["id"]).upper()
        for v in top_validations
    }

    trovati_validi = [
        v
        for v in trovati
        if v in id_validi
    ]

    # Format retrieval results with the number of shared keywords
    stringa_risposta_retrieval = ", ".join(
        [
            f"{v['id']}: ({v['num_comuni']} keywords)"
            for v in top_validations
        ]
    )

    # Store structured results
    if not trovati_validi:
        risultati.append(
            {
                "ID_FAKE": fake_id,
                "RISPOSTA_RETRIEVAL": stringa_risposta_retrieval,
                "ID_VALIDAZIONE": "NONE",
            }
        )
    else:
        risultati.append(
            {
                "ID_FAKE": fake_id,
                "RISPOSTA_RETRIEVAL": stringa_risposta_retrieval,
                "ID_VALIDAZIONE": ", ".join(trovati_validi),
            }
        )


# ============================================================
# FINAL OUTPUT
# ============================================================
df_out = pd.DataFrame(risultati)
df_out.to_excel(OUTPUT_FILE, index=False)

print(f"\nTest completed! File saved to: {OUTPUT_FILE}")

pynvml.nvmlShutdown()