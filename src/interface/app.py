from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder
from llama_cpp import Llama

# ============================================================
# PATH CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

FAKE_PATH = REPO_ROOT / "data" / "FakeCTI.csv"
TUPLES_PATH = REPO_ROOT / "data" / "interface" / "CTI_Tuples.xlsx"
SOURCES_PATH = REPO_ROOT / "data" / "interface" / "Evidence_Sources.xlsx"

MODEL_PATH = SCRIPT_DIR / "Mistral-Nemo-Instruct-2407-Q4_K_S.gguf"

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FakeCTI - Cyber Threat Intelligence",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# THEME / HEADER
# ============================================================

st.title("🛡️ FakeCTI")
st.subheader("Cyber Threat Intelligence — Disinformation Campaign Analysis")

st.markdown(
    """
    Analyze disinformation campaigns using semantic retrieval,
    evidence correlation and LLM-based CTI analysis.
    """
)

st.divider()

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    df_fake = pd.read_csv(
        FAKE_PATH,
        sep=";",
        encoding="latin1"
    )

    df_tuple = pd.read_excel(
        TUPLES_PATH
    )

    df_sources = pd.read_excel(
        SOURCES_PATH
    )

    df_fake.columns = df_fake.columns.str.strip()
    df_tuple.columns = df_tuple.columns.str.strip()
    df_sources.columns = df_sources.columns.str.strip()

    return df_fake, df_tuple, df_sources


df_fake, df_tuple, df_sources = load_data()


# ============================================================
# SOURCE-TO-LINK MAPPING
# ============================================================

source_to_link = dict(
    zip(
        df_sources["FONTE"].astype(str).str.strip(),
        df_sources["LINK"].astype(str).str.strip()
    )
)


def get_evidence_sources(evidence, source_to_link):

    matches = []

    evidence_normalized = " ".join(
        str(evidence).replace("\xa0", " ").split()
    ).lower()

    for source, link in source_to_link.items():

        source_normalized = " ".join(
            str(source).replace("\xa0", " ").split()
        ).lower()

        if source_normalized in evidence_normalized:
            matches.append((source, link))

    return matches

# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():

    # Embedding model
    embedder = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # Cross-Encoder
    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu"
    )

    # Mistral-Nemo
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_gpu_layers=-1,
        n_ctx=4096,
        verbose=False
    )

    return embedder, reranker, llm


with st.spinner("Loading AI models..."):
    embedder, reranker, llm = load_models()

# ============================================================
# DOCUMENT CONSTRUCTION
# ============================================================

def parse_tuple(tuple_text):

    parts = str(tuple_text).split(" - ", 2)

    if len(parts) != 3:
        return None, None, None

    subject = parts[0].strip()
    relation = parts[1].strip()
    object_ = parts[2].strip()

    return subject, relation, object_


@st.cache_data
def build_documents(df_tuple):

    documents = []

    for _, row in df_tuple.iterrows():

        campaign = str(row["CAMPAGNA"]).strip()
        primary_category = str(row["CATEGORIA_PRIMARIA"]).strip()

        secondary_category = (
            ""
            if pd.isna(row["CATEGORIA_SECONDARIA"])
            else str(row["CATEGORIA_SECONDARIA"]).strip()
        )

        evidence = str(row["EVIDENZA"]).strip()
        tuple_text = str(row["TUPLA"]).strip()

        subject, relation, object_ = parse_tuple(tuple_text)

        if subject is None:
            continue

        embedding_text = (
            f"Subject: {subject}. "
            f"Relation: {relation}. "
            f"Object: {object_}."
        )

        document = {
            "campaign": campaign,
            "primary_category": primary_category,
            "secondary_category": secondary_category,
            "evidence": evidence,
            "subject": subject,
            "relation": relation,
            "object": object_,
            "tuple": tuple_text,
            "embedding_text": embedding_text
        }

        documents.append(document)

    return documents


documents = build_documents(df_tuple)

# ============================================================
# TUPLE EMBEDDINGS
# ============================================================

@st.cache_data
def create_embeddings(_embedder, documents):

    texts = [
        doc["embedding_text"]
        for doc in documents
    ]

    embeddings = _embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.asarray(
        embeddings,
        dtype="float32"
    )


with st.spinner("Preparing semantic index..."):
    embeddings = create_embeddings(
        embedder,
        documents
    )

# ============================================================
# DATASET OVERVIEW
# ============================================================

st.success("Datasets loaded successfully.")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Fake News",
        len(df_fake)
    )

with col2:
    st.metric(
        "Tuple",
        len(df_tuple)
    )

st.divider()

# ============================================================
# CAMPAIGNS
# ============================================================

campaigns = sorted(
    df_tuple["CAMPAGNA"]
    .dropna()
    .astype(str)
    .unique()
)

selected_campaign = st.selectbox(
    "🎯 Select Disinformation Campaign",
    campaigns
)

st.write(
    f"Campaign selected: **{selected_campaign}**"
)

# ============================================================
# ANALYTICAL QUESTIONS
# ============================================================

QUESTIONS = {

    "ATTORI / SOGGETTI":
        "Who are the actors or subjects involved in the campaign?",

    "TATTICHE / METODO":
        "What tactics or methods are used in the campaign?",

    "MEZZI / STRUMENTI":
        "What tools or means are used in the campaign?",

    "OBIETTIVO / TARGET":
        "What are the objectives or targets of the campaign?",

    "CONTENUTO / NARRATIVA":
        "What is the main narrative or content promoted by the campaign?",

    "IMPATTO / EFFETTO":
        "What impacts or effects are associated with the campaign?",

    "VALIDAZIONE ANALITICA":
        "What analytical validations provide evidence about the campaign?",

    "CONTESTO / ECOSISTEMA":
        "What context or ecosystem surrounds the campaign?",

    "FINANZIAMENTO / MODELLO ECONOMICO":
        "What information is available about the campaign's funding or economic model?"
}

question_options = list(QUESTIONS.items())

selected_category, selected_question = st.selectbox(
    "🔎 Select Analytical Question",
    question_options,
    format_func=lambda x: f"[{x[0]}] {x[1]}"
)

st.write(
    f"Category: **{selected_category}**"
)

st.write(
    f"Question: **{selected_question}**"
)

st.divider()

# ============================================================
# ANALYSIS
# ============================================================

if st.button(
    "⚡ ANALYZE CAMPAIGN",
    use_container_width=True
):

    # ========================================================
    # STEP 1 — CAMPAIGN FILTERING
    # ========================================================

    campaign_indices = [
        i
        for i, doc in enumerate(documents)
        if doc["campaign"].strip().lower()
        == selected_campaign.strip().lower()
    ]

    if not campaign_indices:

        st.warning(
            "Not enough information: no tuples available "
            "for the selected campaign."
        )

        st.stop()

    st.info(
        f"🔎 {len(campaign_indices)} tuples available "
        f"for **{selected_campaign}**"
    )

    # ========================================================
    # STEP 2 — QUESTION EMBEDDING
    # ========================================================

    with st.spinner("Embedding analytical question..."):

        question_embedding = embedder.encode(
            [selected_question],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        question_embedding = question_embedding.astype(
            "float32"
        )

    # ========================================================
    # STEP 3 — FAISS RETRIEVAL
    # ========================================================

    with st.spinner("Running FAISS semantic retrieval..."):

        campaign_embeddings = embeddings[
            campaign_indices
        ]

        campaign_index = faiss.IndexFlatIP(
            campaign_embeddings.shape[1]
        )

        campaign_index.add(
            campaign_embeddings
        )

        TOP_K = 10

        distances, local_indices = campaign_index.search(
            question_embedding,
            min(TOP_K, len(campaign_indices))
        )

    retrieved_documents = []

    for local_idx, distance in zip(
        local_indices[0],
        distances[0]
    ):

        original_idx = campaign_indices[local_idx]

        doc = documents[original_idx].copy()

        doc["faiss_score"] = float(distance)

        retrieved_documents.append(doc)

    # ========================================================
    # STEP 4 — CROSS-ENCODER
    # ========================================================

    with st.spinner("Running Cross-Encoder reranking..."):

        pairs = [
            [
                selected_question,
                doc["embedding_text"]
            ]
            for doc in retrieved_documents
        ]

        scores = reranker.predict(pairs)

    reranked_documents = []

    for doc, score in zip(
        retrieved_documents,
        scores
    ):

        doc_copy = doc.copy()

        doc_copy["cross_encoder_score"] = float(
            score
        )

        reranked_documents.append(
            doc_copy
        )

    # ========================================================
    # CATEGORY-AWARE RANKING
    # ========================================================

    for doc in reranked_documents:

        primary_match = (
            doc["primary_category"].strip().lower()
            == selected_category.strip().lower()
        )

        secondary_match = (
            doc["secondary_category"].strip().lower()
            == selected_category.strip().lower()
        )

        if primary_match:
            category_priority = 2

        elif secondary_match:
            category_priority = 1

        else:
            category_priority = 0

        doc["category_priority"] = category_priority


    reranked_documents.sort(
        key=lambda x: (
            x["category_priority"],
            x["cross_encoder_score"]
        ),
        reverse=True
    )


    TOP_5 = 5

    top5_documents = reranked_documents[
        :TOP_5
    ]

    # ========================================================
    # STEP 5 — CONTEXT CONSTRUCTION
    # ========================================================

    context_parts = []

    for rank, doc in enumerate(
        top5_documents,
        start=1
    ):

        context_parts.append(
            f"""SOURCE {rank}

Campaign: {doc['campaign']}

Primary category: {doc['primary_category']}

Secondary category: {doc['secondary_category'] if doc['secondary_category'] else 'None'}

Evidence:
{doc['evidence']}

Tuple:
{doc['tuple']}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    # =========================
    # CATEGORY DEFINITIONS
    # =========================

    CATEGORY_DEFINITIONS = {
        "ATTORI / SOGGETTI":
            "Key entities, organizations, or groups behind the operation.",

        "TATTICHE / METODO":
            "Operational techniques, psychological manipulation, or methods used.",

        "MEZZI / STRUMENTI":
            "Platforms, digital tools, bots, or communication channels.",

        "OBIETTIVO / TARGET":
            "Intended victims, demographic groups, or strategic goals.",

        "CONTENUTO / NARRATIVA":
            "Specific content produced, themes, memes, or false narratives pushed.",

        "IMPATTO / EFFETTO":
            "Measurable consequences, social unrest, or behavioral changes.",

        "VALIDAZIONE ANALITICA":
            "How the campaign was debunked, verified, or analyzed by experts.",

        "CONTESTO / ECOSISTEMA":
            "Enabling environmental factors, political climate, or social conditions.",

        "FINANZIAMENTO / MODELLO ECONOMICO":
            "Financial structure, budgets, monetization, or resource allocation."
    }


    # =========================
    # PROMPT
    # =========================

    prompt = f"""
You are a Cyber Threat Intelligence analyst.

Campaign: {selected_campaign}

Analytical category: {selected_category}
Category definition: {CATEGORY_DEFINITIONS[selected_category]}

Question: {selected_question}

Use only the retrieved information below to answer the question.

Prioritize sources where the requested category appears as PRIMARY or SECONDARY.
You may also use relevant information from other categories when it helps answer
the analytical question.

Synthesize the relevant information into a concise answer.
Do not add external knowledge or unsupported information.

Before returning the answer, verify every statement:
- it directly answers the analytical question;
- it is supported by the retrieved evidence;
- the evidence explicitly supports the role, relationship, or conclusion
  expressed in the answer.

Do not infer a role, relationship, or conclusion merely because an entity or
fact is mentioned in the evidence.

Remove irrelevant, unsupported, or inferred statements.

If the retrieved information is insufficient to answer the question, return exactly:
"Not enough information."

Retrieved information:
{context}

Answer concisely:
"""

    # ========================================================
    # STEP 7 — MISTRAL
    # ========================================================

    with st.spinner(
        "Generating CTI analytical response..."
    ):

        response = llm.create_chat_completion(

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            max_tokens=300,
            temperature=0.0,
            top_p=0.9,
            stop=["</s>"]
        )

    answer = response[
        "choices"
    ][0][
        "message"
    ][
        "content"
    ].strip()

    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.subheader("🧠 CTI Analysis")

    st.info(answer)

    # ========================================================
    # RETRIEVED EVIDENCE
    # ========================================================

    with st.expander(
        "🔍 View retrieved evidence"
    ):

        for rank, doc in enumerate(
            top5_documents,
            start=1
        ):

            st.markdown(
                f"### Source {rank}"
            )

            st.write(
                f"**Primary category:** {doc['primary_category']}"
            )

            st.write(
                f"**Secondary category:** "
                f"{doc['secondary_category'] if doc['secondary_category'] else 'None'}"
            )

            st.write(
                f"**Tuple:** {doc['tuple']}"
            )

            st.write(
                f"**Evidence:** {doc['evidence']}"
            )

            sources = get_evidence_sources(
                doc["evidence"],
                source_to_link
            )

            if sources:

                st.write("**Original sources:**")

                for source, link in sources:
                    st.markdown(
                        f"- [{source}]({link})"
                    )

            st.caption(
                f"FAISS: {doc['faiss_score']:.4f} | "
                f"Cross-Encoder: {doc['cross_encoder_score']:.4f} | "
                f"Category priority: {doc['category_priority']}"
            )

            st.divider()