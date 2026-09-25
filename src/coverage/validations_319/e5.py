from pathlib import Path
import re
import time

import pandas as pd
import pynvml
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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
FILE_VALIDAZIONI = DATA_DIR / "CTI_Evidences_319.xlsx"
FILE_FAKE = DATA_DIR / "Fake101_Benchmark.xlsx"
OUTPUT_FILE = SCRIPT_DIR / "E5_Fake101.xlsx"

TOP_K_RETRIEVAL = 10
MAX_CONTEXT = 3700
SAFETY_MARGIN = 100


# ============================================================
# E5 AND LLM MODEL LOADING
# ============================================================
print("Loading multilingual-e5-large model...")
sbert_model = SentenceTransformer(
    "intfloat/multilingual-e5-large",
    device="cpu",
)
print(">>> E5 MODEL LOADED")

print("Loading Mistral model on GPU...")
llm = Llama(
    model_path=str(MODEL_PATH),
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=False,
)
print(">>> LLM LOADED")


# ============================================================
# CTI EVIDENCE CORPUS LOADING AND EMBEDDING
# ============================================================
df_val = pd.read_excel(FILE_VALIDAZIONI)
df_val.columns = df_val.columns.str.strip()

validazioni_corpus = []
corpus_texts = []

for _, row in df_val.iterrows():
    v_id = str(row["ID"]).strip()
    v_campagna = str(row["CAMPAGNA"]).strip()
    v_testo = str(row["EVIDENZA"]).strip()

    validazioni_corpus.append(
        {
            "id": v_id,
            "campagna": v_campagna,
            "testo": v_testo,
        }
    )

    # E5 document prefix
    corpus_texts.append(f"passage: {v_testo}")

print(f">>> CTI evidence corpus loaded: {len(validazioni_corpus)} elements")

print("Generating embeddings for the CTI evidence corpus...")

corpus_embeddings = sbert_model.encode(
    corpus_texts,
    show_progress_bar=True,
    normalize_embeddings=True,
)

print(">>> CORPUS INDEXED WITH E5")


# ============================================================
# E5 RETRIEVAL WITH COSINE SIMILARITY
# ============================================================
def sbert_retrieve(fake_text, top_k=TOP_K_RETRIEVAL):
    # E5 query prefix
    query_text = f"query: {fake_text}"

    fake_embedding = sbert_model.encode(
        [query_text],
        normalize_embeddings=True,
    )

    scores = cosine_similarity(
        fake_embedding,
        corpus_embeddings,
    ).flatten()

    risultati = []

    for idx, score in enumerate(scores):
        val = validazioni_corpus[idx]

        risultati.append(
            {
                **val,
                "total_score": float(score),
                "sbert_score": float(score),
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
You are an expert Cyber Threat Intelligence (CTI) and Disinformation analyst.

Your task is to select, from the candidate validations, ONLY those that provide evidence that DIRECTLY or INDIRECTLY debunks, refutes, exposes, or analytically undermines a specific claim, source, actor, or documented operation behind the FAKE NEWS below — not evidence that merely shares its general topic or campaign.

FAKE NEWS:
ID: {fake_id}
Campaign: {campaign}
Text: {fake_text_placeholder}

CANDIDATE VALIDATIONS:
{validation_text}

CORE TEST:
"Does this validation provide concrete evidence about the SAME SPECIFIC CLAIM made by the fake, or about the specific source, article, event, or documented operation responsible for producing it?"

SELECT a validation if at least ONE of the following conditions holds:

1. SAME SPECIFIC CLAIM
The validation provides factual or analytical evidence that directly contradicts, disproves, debunks, or exposes as false or misleading the specific proposition asserted by the fake.
The validation does NOT need to refer to the exact same article, wording, publication, or instance if it clearly addresses the same specific factual claim.
Different wording is acceptable when the underlying factual proposition is the same.
However, sharing only the same general subject is NOT sufficient.
Example:
Fake: "Vaccine X causes infertility."
Validation: "A study found no association between Vaccine X and infertility."
→ SELECT: same specific factual claim.
Validation: "Vaccines are generally considered safe."
→ REJECT: same broad topic, but it does not address the specific claim.

2. SAME SPECIFIC INSTANCE
The validation explicitly concerns the same article, fabricated story, event, document, image, video, website, account, source, or documented disinformation operation responsible for THIS fake.
A shared actor, organization, website, source, or campaign alone is NOT sufficient.
If the same actor or source is associated with multiple different fake stories or operations, the validation must establish a specific connection between that actor/source and THIS fake.
The validation does not need to repeat the entire fake. It is sufficient if it clearly identifies or exposes the specific fabricated story, source, or operation from which the fake originates.

3. SPECIFIC DEBUNKING CONNECTION
The validation provides concrete information that directly undermines or exposes a specific factual element, source, fabrication, manipulation, or mechanism explicitly present in THIS fake.
An evidence does NOT need to refute every statement in the fake. It may debunk or expose one specific and identifiable part of it.
However, the connection must be supported by the candidate validation itself.
Do NOT infer that a tactic, actor, narrative, source, or operation applies to the fake merely because such a connection would be plausible.

RELEVANCE TEST:
For EVERY candidate validation, ask:
"What specific claim or concrete element of THIS fake does this validation help refute, expose, debunk, or analytically undermine?"
If you can identify a concrete answer supported by the validation itself, the evidence may be selected.
If the answer is only based on general similarity, thematic overlap, campaign membership, shared keywords, or plausible inference, REJECT it.

REJECT a validation if the connection exists only at the level of:
- same broad topic;
- same campaign label;
- same country;
- same election or political context;
- same general narrative;
- same keywords;
- same general actor without a demonstrated connection to this fake;
- same general source without a demonstrated connection to this fake;
- same general disinformation tactic;
- same type of fake;
- general CTI or historical background;
- general information about disinformation operations.

Sharing a campaign, actor, country, election year, political context, narrative, or broad topic is NEVER sufficient by itself.

DEFAULT RULE:
Assume that a candidate validation is NOT relevant.
Select it only when there is a clear and concrete connection to a specific claim or element of THIS fake, or to the specific documented source or operation responsible for it.
When the relationship requires unsupported assumptions or inference, REJECT it.
Most fake news items may have NO valid evidence among the candidates.
If no candidate clearly satisfies the criteria, return NONE.

REAL FAILURE PATTERNS TO AVOID:
Mistake 1:
A validation about hoax author "Paul Horner" and a specific hoax article he wrote gets wrongly selected for OTHER 2016-election fakes that are not that specific Horner article.
→ Only select Horner-related validations when the fake is the specific Horner hoax documented by the validation, or when the validation establishes a concrete connection between Horner and THIS fake.
For example, if the fake describes the specific fabricated story about protesters being paid and the validation documents Horner as the author/source of that exact fabricated story:
→ SELECT.
If the fake is merely another Trump/election fake with no demonstrated connection to Horner:
→ REJECT.
Do NOT become reluctant to select Horner-related evidence altogether. Apply the same CORE TEST as for every other candidate.

Mistake 2:
A validation citing general historical statistics on foreign election interference (e.g. "the US/USSR intervened in N elections between 1946-2000") or general findings about Russian-funded influence operations gets wrongly selected for ANY fake that mentions Trump, Clinton, Russia, or the election in passing — even purely satirical fakes involving celebrities, fictional characters, or unrelated political stories.
→ These validations are background statistics or general findings about the phenomenon.
Select them ONLY when they concretely address a specific claim made by the fake or establish a documented connection between THIS fake and the interference operation.
Political or thematic similarity alone is NOT sufficient.

Mistake 3:
A validation about Russian funding of influencers or Putin directing 2016/2024 influence operations gets wrongly selected for fakes that only mention Russia, Trump, Clinton, Putin, or elections in passing.
→ Only select if the validation provides a concrete connection between THIS fake, its specific claim, its source, or its documented operation and that funding/directive.
The presence of the same country, political actor, or election is NOT sufficient.

GENERAL PRINCIPLE BEHIND ALL THREE MISTAKES:
The relevant unit is the SPECIFIC CLAIM or SPECIFIC DOCUMENTED INSTANCE — not the broad topic.
Same specific factual claim → potentially SELECT.
Same specific article/story/source/operation → potentially SELECT.
Specific evidence undermining a concrete element of the fake → potentially SELECT.
Same broad topic/campaign/actor/country/tactic only → REJECT.

FINAL CHECK BEFORE SELECTING EACH ID:
Ask these questions:
1. What exact claim or concrete element of the fake does this validation address?
2. Is that connection explicitly supported by the validation, rather than inferred?
3. Would I still select this validation for many unrelated fake stories in the same campaign simply because they share the same topic, actor, or political context?
If the answer to question 1 is unclear → REJECT.
If question 2 requires unsupported inference → REJECT.
If the answer to question 3 is YES → the validation is probably too generic → REJECT.

OUTPUT FORMAT:
Do NOT write conversational introductions, reasoning, or explanations.
Return ONLY the matching IDs separated by commas (e.g., CT201, AN199, TC200) or NONE.
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

    # Retrieve the Top-K candidate evidences
    print(
        f"\nE5 retrieval "
        f"(TOP {TOP_K_RETRIEVAL} candidates)..."
    )

    top_validations = sbert_retrieve(fake_text)

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
            max_tokens=50,
        )

        raw_output = (
            response["choices"][0]["message"]["content"].strip()
        )

    except Exception as e:
        print(f"MODEL ERROR: {e}")
        raw_output = "NONE"

    print(f"\n>>> MODEL RESPONSE: {raw_output}")

    # Extract and validate alphanumeric evidence IDs
    trovati = re.findall(
        r"\b[A-Z]{1,2}\d{3}\b",
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

    # Format retrieval results with E5 similarity scores
    stringa_risposta_retrieval = ", ".join(
        [
            f"{v['id']}: (score: {v['sbert_score']:.3f})"
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