import os
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer

from pipeline import (
    get_fake_by_id,
    build_tfidf_index,
    build_keyword_index,
    analyze_fake,
    generate_explanation
)


# ============================================================
# PATH
# ============================================================

FAKE_PATH = "FakeCTI.csv"

VALIDATIONS_90_PATH = (
    "elenco_validazione_analitica_doppia.xlsx"
)

EVIDENCES_319_PATH = (
    "elenco_evidenze_codificate.xlsx"
)

SAMPLE_PATH = "sample_1000.xlsx"

OUTPUT_PATH = "pipeline_results_1000_Gemma4_31B.xlsx"

API_URL = os.getenv(
    "LLM_API_URL",
    "https://openrouter.ai/api/v1/chat/completions"
)
API_KEY = os.getenv("LLM_API_KEY")
MODEL_NAME = os.getenv(
    "LLM_MODEL_NAME",
    "google/gemma-4-31b-it"
)
API_TIMEOUT = 300

# ============================================================
# API
# ============================================================

class OpenAICompatibleClient:

    def __init__(self, api_url, api_key, model_name, timeout=300):
        if not api_key:
            raise RuntimeError(
                "LLM_API_KEY non impostata. Configurare la chiave API prima dell'esecuzione."
            )
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout

    def call(self, prompt, max_tokens, temperature=0.0, reasoning=False):
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "reasoning": {"enabled": reasoning}
        }

        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=self.timeout
        )

        if not response.ok:
            print(f"\nAPI ERROR {response.status_code}:")
            print(response.text)

        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}

        return {
            "content": content or "",
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0)
        }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CARICAMENTO DATI")
print("=" * 70)

print("Caricamento FakeCTI...")

fake_df = pd.read_csv(
    FAKE_PATH,
    sep=";",
    encoding="latin1"
)


print("Caricamento 90 validazioni...")

validations_90_df = pd.read_excel(
    VALIDATIONS_90_PATH
)


print("Caricamento 319 evidenze...")

evidences_319_df = pd.read_excel(
    EVIDENCES_319_PATH
)

sample_df = pd.read_excel(
    SAMPLE_PATH,
    sheet_name="SAMPLE"
)


# ============================================================
# CONTROLLI CAMPIONE
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
        f"Colonne mancanti in sample.xlsx: "
        f"{missing_columns}"
    )


if sample_df["ID_FAKE"].duplicated().any():
    raise ValueError(
        "Sono presenti ID_FAKE duplicati nel campione."
    )


print(
    f"\nFake presenti nel campione: "
    f"{len(sample_df)}"
)

print(
    f"Campagne presenti: "
    f"{sample_df['CAMPAIGN'].nunique()}"
)


# ============================================================
# BUILD RETRIEVAL INDEXES
# ============================================================

print("\n" + "=" * 70)
print("COSTRUZIONE INDICI")
print("=" * 70)


print(
    "Costruzione indice TF-IDF "
    "sulle 90 validazioni..."
)

tfidf_vectorizer, tfidf_matrix = (
    build_tfidf_index(
        validations_90_df
    )
)


print(
    "Costruzione indice Keyword "
    "sulle 319 evidenze..."
)

keyword_corpus = build_keyword_index(
    evidences_319_df
)


print("Indici retrieval pronti.")


# ============================================================
# LOAD E5
# ============================================================

print("\n" + "=" * 70)
print("CARICAMENTO E5")
print("=" * 70)

e5_model = SentenceTransformer(
    "intfloat/multilingual-e5-large",
    device="cpu"
)

print("E5 caricato.")


# ============================================================
# LOAD LLM API CLIENT
# ============================================================

print("\n" + "=" * 70)
print("CARICAMENTO GEMMA 4 31B")
print("=" * 70)

llm = OpenAICompatibleClient(
    api_url=API_URL,
    api_key=API_KEY,
    model_name=MODEL_NAME,
    timeout=API_TIMEOUT
)

print(f"Modello configurato: {MODEL_NAME}")
print(f"Endpoint API: {API_URL}")

# ============================================================
# GROUNDING CONTEXT PER IL JUDGE
# ============================================================

def get_grounding_context(
    stage,
    selected_ids,
    pattern,
    validations_90_df,
    evidences_319_df
):
    """
    Ricostruisce il contesto utilizzato per generare
    la spiegazione finale.

    Stage 1 -> validazioni analitiche selezionate
    Stage 2 -> evidenze CTI selezionate
    Stage 3 -> pattern di campagna
    """

    # Stage 3: il grounding è direttamente il pattern
    if stage == 3:
        return pattern if pattern else ""

    if not selected_ids:
        return ""

    # Dataset da utilizzare
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

if os.path.exists(OUTPUT_PATH):

    print("\n" + "=" * 70)
    print("CHECKPOINT TROVATO")
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
        f"Fake già completate: "
        f"{len(completed_ids)}"
    )

    results = results_df.to_dict(
        orient="records"
    )

else:

    print("\nNessun checkpoint precedente.")

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
    # SKIP SE GIÀ COMPLETATA
    # --------------------------------------------------------

    if fake_id in completed_ids:

        print(
            f"\n[{sample_index}/{total}] "
            f"{fake_id} già completata -> SKIP"
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
        # RECUPERA FAKE DAL DATASET ORIGINALE
        # ====================================================

        fake = get_fake_by_id(
            fake_df,
            fake_id
        )


        print(
            f"Campaign: "
            f"{fake['campaign']}"
        )


        # ====================================================
        # ANALISI 3-STAGE
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
        # GENERAZIONE SPIEGAZIONE
        # ====================================================

        explanation = generate_explanation(
            fake=fake,
            result=result,
            llm=llm
        )

        # ====================================================
        # PREPARA DATI RISULTATO
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
        # GROUNDING CONTEXT PER IL JUDGE
        # ====================================================

        grounding_context = get_grounding_context(
            stage=result["stage"],
            selected_ids=selected_ids,
            pattern=pattern,
            validations_90_df=validations_90_df,
            evidences_319_df=evidences_319_df
        )


        # ====================================================
        # SALVA RISULTATO
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
            f"Tipo: "
            f"{result['source_type']}"
        )

        if result["stage"] in [1, 2]:

            print(
                f"Evidenze: "
                f"{selected_ids_text}"
            )

        elif result["stage"] == 3:

            print(
                "\nPattern:"
            )

            print(pattern)


        print(
            "\nSpiegazione:"
        )

        print(explanation)


    # ========================================================
    # GESTIONE ERRORE
    # ========================================================

    except Exception as e:

        print(
            f"\nERRORE SU {fake_id}: "
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
    # CHECKPOINT DOPO OGNI FAKE
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
        f"\nCheckpoint salvato -> "
        f"{OUTPUT_PATH}"
    )


# ============================================================
# FINE
# ============================================================

print("\n" + "=" * 70)
print("BATCH COMPLETATO")
print("=" * 70)


results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "SAMPLE_INDEX"
)


# ============================================================
# STATISTICHE FINALI
# ============================================================

ok_df = results_df[
    results_df["STATUS"] == "OK"
]

error_df = results_df[
    results_df["STATUS"] == "ERROR"
]


print(
    f"Fake completate correttamente: "
    f"{len(ok_df)}"
)

print(
    f"Fake con errore: "
    f"{len(error_df)}"
)


if len(ok_df) > 0:

    print(
        "\nDistribuzione Stage:"
    )

    print(
        ok_df["STAGE"]
        .value_counts()
        .sort_index()
    )

print(
    f"\nRisultati salvati in:\n"
    f"{OUTPUT_PATH}"
)