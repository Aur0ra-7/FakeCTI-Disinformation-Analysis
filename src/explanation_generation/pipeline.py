import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K_TASK1 = 10
TOP_K_TASK2 = 10
TOP_K_TASK3 = 10

TASK1_MAX_CONTEXT = 4096
TASK1_SAFETY_MARGIN = 200

TASK2_MAX_CONTEXT = 3700
TASK2_SAFETY_MARGIN = 100

TASK3_MAX_CONTEXT = 4096
TASK3_SAFETY_MARGIN = 200

FEATURE_MAX_OUTPUT_TOKENS = 300
PATTERN_MAX_OUTPUT_TOKENS = 300

FAKE_ID_COL = "ID"
FAKE_TEXT_COL = "TEXT"
FAKE_CAMPAIGN_COL = "CAMPAIGN"

EVIDENCE_ID_COL = "ID"
EVIDENCE_TEXT_COL = "EVIDENZA"


# ============================================================
# UTILITIES
# ============================================================

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def strip_text(text):
    if pd.isna(text):
        return ""
    return str(text).strip()

def format_evidence(evidence):

    blocks = []

    for item in evidence:
        blocks.append(
            f"ID: {item['id']}\n"
            f"Evidence: {item['text']}"
        )

    return "\n\n---\n\n".join(blocks)

def normalize_fake_id(fake_id):
    """
    Accepts:
        F48
        f48
        48

    Returns:
        48
    """

    fake_id = str(fake_id).strip().upper()

    if fake_id.startswith("F"):
        fake_id = fake_id[1:]

    return fake_id


def display_fake_id(fake_id):
    """
    48 -> F48
    """

    return f"F{normalize_fake_id(fake_id)}"


def truncate_fake_for_prompt(
    llm,
    fake,
    candidates,
    prompt_builder,
    max_context,
    safety_margin
):

    fake_placeholder = fake.copy()
    fake_placeholder["text"] = "[PLACEHOLDER]"

    prompt_base = prompt_builder(
        fake_placeholder,
        candidates
    )

    base_tokens = len(
        llm.tokenize(
            prompt_base.encode("utf-8")
        )
    )

    max_allowed_fake = (
        max_context
        - base_tokens
        - safety_margin
    )

    fake_tokens = llm.tokenize(
        fake["text"].encode("utf-8")
    )

    if len(fake_tokens) > max_allowed_fake:

        truncated_text = llm.detokenize(
            fake_tokens[:max_allowed_fake]
        ).decode(
            "utf-8",
            errors="ignore"
        )

    else:
        truncated_text = fake["text"]

    final_fake = fake.copy()
    final_fake["text"] = truncated_text

    return final_fake


def truncate_fake_for_generation(
    llm,
    fake,
    prompt_builder,
    max_context,
    max_output_tokens,
    safety_margin,
    **builder_kwargs
):
    fake_placeholder = fake.copy()
    fake_placeholder["text"] = ""

    base_prompt = prompt_builder(
        fake=fake_placeholder,
        **builder_kwargs
    )

    base_tokens = len(
        llm.tokenize(
            base_prompt.encode("utf-8")
        )
    )

    available_fake_tokens = (
        max_context
        - base_tokens
        - max_output_tokens
        - safety_margin
    )

    if available_fake_tokens <= 0:
        raise RuntimeError(
            "The generation prompt exceeds the context window "
            "even without the fake-news text."
        )

    fake_tokens = llm.tokenize(
        fake["text"].encode("utf-8")
    )

    if len(fake_tokens) <= available_fake_tokens:
        return fake

    truncated_fake = fake.copy()

    truncated_fake["text"] = llm.detokenize(
        fake_tokens[:available_fake_tokens]
    ).decode(
        "utf-8",
        errors="ignore"
    )

    print(
        f"\nFake-news text truncated for generation: "
        f"{len(fake_tokens)} -> {available_fake_tokens} token"
    )

    return truncated_fake

# ============================================================
# FAKE NEWS RETRIEVAL
# ============================================================

def get_fake_by_id(fake_df, fake_id):

    numeric_id = normalize_fake_id(fake_id)

    ids = (
        fake_df[FAKE_ID_COL]
        .astype(str)
        .str.strip()
    )

    match = fake_df[ids == numeric_id]

    if match.empty:
        raise ValueError(
            f"Fake news {display_fake_id(fake_id)} not found in FakeCTI.csv."
        )

    row = match.iloc[0]

    return {
        "id": display_fake_id(numeric_id),
        "original_id": numeric_id,
        "campaign": strip_text(row[FAKE_CAMPAIGN_COL]),
        "text": strip_text(row[FAKE_TEXT_COL]),
    }


# ============================================================
# MISTRAL OUTPUT PARSING
# ============================================================

def parse_model_output(output, candidates):

    if output is None:
        return []

    output = output.strip().upper()

    if not output or output == "NONE":
        return []

    selected_ids = []

    for candidate in candidates:

        candidate_id = str(
            candidate["id"]
        ).strip().upper()

        if re.search(
            rf"(?<!\w){re.escape(candidate_id)}(?!\w)",
            output
        ):
            selected_ids.append(candidate_id)

    return selected_ids


# ============================================================
# CANDIDATE FORMATTING
# ============================================================

def format_task1_candidates(candidates):

    blocks = []

    for c in candidates:
        blocks.append(
            f"{c['id']}: {c['text']}"
        )

    return "\n\n".join(blocks)


def format_task2_candidates(candidates):

    blocks = []

    for c in candidates:
        blocks.append(
            f"ID: {c['id']}\n"
            f"Campaign: {c['campaign']}\n"
            f"Validation: {c['text']}\n"
            f"---"
        )

    return "\n".join(blocks)


# ============================================================
# STAGE 1 RETRIEVAL
# TF-IDF OVER THE 90 ANALYTICAL VALIDATIONS
# ============================================================

TFIDF_STOPWORDS = [
    "the", "a", "an", "and", "or", "but",
    "in", "on", "at", "to", "for", "with",
    "by", "of", "is", "are", "was", "were",
    "this", "that", "it", "as", "from"
]


def build_tfidf_index(evidence_df):
    """
    Build the TF-IDF index over the 90 analytical validations once.
    """

    texts = (
        evidence_df[EVIDENCE_TEXT_COL]
        .fillna("")
        .astype(str)
        .tolist()
    )

    vectorizer = TfidfVectorizer(
        stop_words=TFIDF_STOPWORDS,
        lowercase=True
    )

    evidence_matrix = vectorizer.fit_transform(texts)

    return vectorizer, evidence_matrix


def tfidf_retrieval(
    fake_text,
    evidence_df,
    vectorizer,
    evidence_matrix,
    top_k=10
):
    """
    TF-IDF retrieval used in the coverage experiment.
    """

    fake_vector = vectorizer.transform([fake_text])

    scores = cosine_similarity(
        fake_vector,
        evidence_matrix
    ).flatten()

    ranking = scores.argsort()[::-1][:top_k]

    results = []

    for idx in ranking:

        row = evidence_df.iloc[idx]

        results.append({
            "id": clean_text(row[EVIDENCE_ID_COL]),
            "campaign": clean_text(row["CAMPAGNA"]),
            "text": clean_text(row[EVIDENCE_TEXT_COL]),
            "score": float(scores[idx])
        })

    return results


# ============================================================
# STAGE 2 RETRIEVAL
# KEYWORD RETRIEVAL OVER THE 319 CTI EVIDENCES
# ============================================================

KEYWORD_STOPWORDS = {
    "the", "a", "an", "and", "or", "but",
    "in", "on", "at", "to", "for", "with",
    "by", "of", "is", "are", "was", "were",
    "this", "that", "it", "as", "from"
}


def build_keyword_index(evidence_df):
    """
    Precompute keywords for the 319 CTI evidences.
    """

    corpus = []

    for _, row in evidence_df.iterrows():

        evidence_text = strip_text(
        row[EVIDENCE_TEXT_COL]
    )

        words = re.findall(
            r"\b\w+\b",
            evidence_text.lower()
        )

        keywords = {
            word
            for word in words
            if len(word) > 3
            and word not in KEYWORD_STOPWORDS
        }

        corpus.append({
            "id": strip_text(row[EVIDENCE_ID_COL]),
            "campaign": strip_text(row["CAMPAGNA"]),
            "text": evidence_text,
            "keywords": keywords
        })

    return corpus


def keyword_retrieval(
    fake_text,
    keyword_corpus,
    top_k=10
):
    """
    Keyword retrieval used in the coverage experiment.
    """

    fake_words = re.findall(
        r"\b\w+\b",
        strip_text(fake_text).lower()
    )

    fake_keywords = {
        word
        for word in fake_words
        if len(word) > 3
        and word not in KEYWORD_STOPWORDS
    }

    results = []

    for evidence in keyword_corpus:

        common = fake_keywords.intersection(
            evidence["keywords"]
        )

        score = len(common)

        results.append({
            "id": evidence["id"],
            "campaign": evidence["campaign"],
            "text": evidence["text"],
            "score": score,
            "matched_keywords": sorted(common)
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# E5 CACHE BY CAMPAIGN
# ============================================================

_E5_CAMPAIGN_CACHE = {}


# ============================================================
# STAGE 3 RETRIEVAL
# E5 WITHIN THE QUERY CAMPAIGN ONLY
# ============================================================

def campaign_fake_retrieval(
    fake,
    fake_df,
    e5_model,
    top_k=10
):

    campaign = fake["campaign"]

    # --------------------------------------------------------
    # 1. COMPLETE CAMPAIGN CACHE
    # --------------------------------------------------------

    if campaign not in _E5_CAMPAIGN_CACHE:

        campaign_mask = (
            fake_df[FAKE_CAMPAIGN_COL]
            .fillna("")
            .astype(str)
            .str.strip()
            == campaign
        )

        campaign_df = (
            fake_df[campaign_mask]
            .copy()
        )

        # Clean texts
        campaign_df["_TEXT_CLEAN"] = (
            campaign_df[FAKE_TEXT_COL]
            .apply(strip_text)
        )

        campaign_df = (
            campaign_df[
                campaign_df["_TEXT_CLEAN"] != ""
            ]
            .copy()
            .reset_index(drop=True)
        )

        if campaign_df.empty:
            return []

        passages = [
            f"passage: {text}"
            for text in campaign_df["_TEXT_CLEAN"].tolist()
        ]

        print(
            f"\nGenerating E5 embeddings "
            f"for campaign '{campaign}' "
            f"({len(campaign_df)} fake news items)..."
        )

        campaign_embeddings = e5_model.encode(
            passages,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        _E5_CAMPAIGN_CACHE[campaign] = {
            "df": campaign_df,
            "embeddings": campaign_embeddings
        }

    else:

        print(
            f"\nE5 embeddings for campaign "
            f"'{campaign}' retrieved from cache."
        )

    # --------------------------------------------------------
    # 2. CACHE RECOVERY
    # --------------------------------------------------------

    cached = _E5_CAMPAIGN_CACHE[campaign]

    campaign_df = cached["df"]
    campaign_embeddings = cached["embeddings"]

    # --------------------------------------------------------
    # 3. QUERY EXCLUSION
    # --------------------------------------------------------

    campaign_ids = (
        campaign_df[FAKE_ID_COL]
        .astype(str)
        .str.strip()
    )

    candidate_mask = (
        campaign_ids != fake["original_id"]
    )

    candidate_df = (
        campaign_df[candidate_mask]
        .copy()
        .reset_index(drop=True)
    )

    candidate_embeddings = (
        campaign_embeddings[candidate_mask.to_numpy()]
    )

    if candidate_df.empty:
        return []

    print(
        f"\nCandidate fake news in campaign "
        f"'{campaign}': {len(candidate_df)}"
    )

    # --------------------------------------------------------
    # 4. QUERY E5
    # --------------------------------------------------------

    query_embedding = e5_model.encode(
        [f"query: {fake['text']}"],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # --------------------------------------------------------
    # 5. SIMILARITY
    # --------------------------------------------------------

    scores = cosine_similarity(
        query_embedding,
        candidate_embeddings
    ).flatten()

    # --------------------------------------------------------
    # 6. RESULTS
    # --------------------------------------------------------

    results = []

    for idx, score in enumerate(scores):

        row = candidate_df.iloc[idx]

        results.append({
            "id": display_fake_id(
                row[FAKE_ID_COL]
            ),
            "campaign": strip_text(
                row[FAKE_CAMPAIGN_COL]
            ),
            "text": strip_text(
                row["_TEXT_CLEAN"]
            ),
            "score": float(score)
        })

    # Stable descending sort
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# STAGE 3A - FEATURE EXTRACTION
# ============================================================

def build_feature_extraction_prompt(fake_id, campaign, fake_text):

    return f"""
You are an expert Disinformation and Cyber Threat Intelligence analyst.

Your task is to extract the characteristics of the following fake news item
that are useful for comparing it with other fake news from the same
disinformation campaign.

FAKE NEWS:
ID: {fake_id}
Campaign: {campaign}
Text: {fake_text}

Extract only information directly supported by the text.

IMPORTANT:
Describe the content and construction of the fake news item, not whether
its claims are factually true or false.

Do not infer that a claim, attribution, source, quote, event, or statement
is false, fabricated, manipulated, biased, unverified, or deceptive unless
the text itself provides information supporting that characterization.

Do not infer strategic intent, persuasion goals, political objectives, or
campaign objectives beyond what is directly supported by the text.

Use this exact structure:

CLAIM_NARRATIVE:
<main claim, storyline, or narrative expressed in the text>

FABRICATION_TYPE:
<observable type of content construction, such as an attributed statement,
reported event, allegation, comparison, causal claim, endorsement, prediction,
or other form directly identifiable from the text>

ACTORS_RELATIONSHIPS:
<main actors or entities and the relationships explicitly presented between them>

FRAMING:
<how actors, organizations, products, or events are portrayed in the text>

CONSTRUCTION_MECHANISM:
<how the narrative is constructed or presented, using only mechanisms
observable in the text>

TARGET_EFFECT:
<target and/or apparent effect only when supported by the text; otherwise
state "Not explicit in the text">

Be concise.
Use only one short sentence per field.
Keep only the main information needed to characterize how this fake news
story is constructed.
Avoid background details, long lists, examples, and repetition.
Keep the entire response approximately within 120-150 tokens when possible.
Do not use Markdown formatting, bold text, italics, or asterisks in the output.

Do NOT determine whether this fake belongs to a recurring pattern.
Do NOT compare it with other fake news.
Do NOT use external knowledge.
Do NOT invent information that is not supported by the text.
""".strip()


def build_feature_prompt_with_truncation(
    llm,
    fake_id,
    campaign,
    fake_text
):

    empty_prompt = build_feature_extraction_prompt(
        fake_id=fake_id,
        campaign=campaign,
        fake_text=""
    )

    instruction_tokens = len(
        llm.tokenize(
            empty_prompt.encode("utf-8")
        )
    )

    available_fake_tokens = (
        TASK3_MAX_CONTEXT
        - instruction_tokens
        - FEATURE_MAX_OUTPUT_TOKENS
        - TASK3_SAFETY_MARGIN
    )

    if available_fake_tokens <= 0:
        return None

    fake_tokens_full = llm.tokenize(
        fake_text.encode("utf-8")
    )

    original_tokens = len(fake_tokens_full)

    if original_tokens > available_fake_tokens:
        fake_tokens_final = fake_tokens_full[
            :available_fake_tokens
        ]
    else:
        fake_tokens_final = fake_tokens_full

    final_fake_text = llm.detokenize(
        fake_tokens_final
    ).decode(
        "utf-8",
        errors="ignore"
    )

    final_prompt = build_feature_extraction_prompt(
        fake_id=fake_id,
        campaign=campaign,
        fake_text=final_fake_text
    )

    return {
        "prompt": final_prompt,
        "original_tokens": original_tokens,
        "final_tokens": len(fake_tokens_final),
        "truncated": len(fake_tokens_final) < original_tokens
    }


def extract_fake_features(
    llm,
    fake_id,
    campaign,
    fake_text
):

    prompt_data = build_feature_prompt_with_truncation(
        llm=llm,
        fake_id=fake_id,
        campaign=campaign,
        fake_text=fake_text
    )

    if prompt_data is None:
        raise RuntimeError(
            f"Unable to build the Feature Extraction prompt "
            f"for {fake_id}."
        )

    prompt = prompt_data["prompt"]

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=FEATURE_MAX_OUTPUT_TOKENS
    )

    features = (
        response["choices"][0]["message"]["content"]
        .strip()
    )

    # Remove any Markdown formatting produced by Mistral
    features = features.replace("*", "")

    return {
        "id": fake_id,
        "features": features,
        "truncated": prompt_data["truncated"]
    }


# ============================================================
# STAGE 3B - PATTERN EXTRACTION
# ============================================================

def build_pattern_extraction_prompt(
    query_id,
    query_features,
    candidate_features
):

    candidates_text = ""

    for candidate in candidate_features:
        candidates_text += (
            f"\nID: {candidate['id']}\n"
            f"{candidate['features']}\n"
            f"---\n"
        )

    return f"""
You are an expert Disinformation and Cyber Threat Intelligence analyst.

Your task is to identify a recurring disinformation pattern for the QUERY FAKE
using fake news retrieved from the same campaign.

The retrieved items were selected by semantic similarity and may contain noise.

QUERY FAKE:
ID: {query_id}
{query_features}

RETRIEVED FAKE NEWS:
{candidates_text}

Use the QUERY FAKE as the anchor.

First identify HOW the QUERY FAKE specifically constructs its false or
misleading narrative. Focus primarily on:
- CLAIM_NARRATIVE;
- FABRICATION_TYPE;
- CONSTRUCTION_MECHANISM.

Use ACTORS_RELATIONSHIPS, FRAMING, and TARGET_EFFECT only as additional
supporting characteristics.

Then determine which retrieved fake news reproduce the SAME underlying
construction.

The pattern must be based on a specific intersection between characteristics
of the QUERY FAKE and characteristics recurring in MULTIPLE retrieved items.

Generic characteristics such as political targets, negative framing,
fabricated content, discrediting, confusion, polarization, or influencing
voters are NOT sufficient to establish a pattern by themselves.

Do NOT:
- derive characteristics from retrieved items that are not present in the QUERY;
- broaden the pattern to include more items;
- combine different constructions using "X or Y";
- include items that match only generic or partial characteristics.

Different actors, events, or claims are allowed when the same underlying
narrative or fabrication mechanism is reproduced.

Every RECURRING ELEMENT must be present in the QUERY FAKE and in EVERY
fake news listed under SUPPORTING FAKE NEWS.

Do not include an element if it is present only in some supporting items.

Do not use Markdown formatting, bold text, italics, or asterisks in the output.

Return exactly:

PATTERN:
<a concise description of the recurring construction>

RECURRING ELEMENTS:
- <specific shared element>
- <specific shared element>
- <specific shared element>

SUPPORTING FAKE NEWS:
<ID>, <ID>, ...

Do NOT include the QUERY FAKE ID in SUPPORTING FAKE NEWS.

If no specific construction of the QUERY FAKE recurs in MULTIPLE retrieved
items, return:

NONE
""".strip()


def extract_pattern(
    llm,
    query_id,
    query_features,
    candidate_features
):

    prompt = build_pattern_extraction_prompt(
        query_id=query_id,
        query_features=query_features,
        candidate_features=candidate_features
    )

    prompt_tokens = len(
        llm.tokenize(
            prompt.encode("utf-8")
        )
    )

    required_tokens = (
        prompt_tokens
        + PATTERN_MAX_OUTPUT_TOKENS
        + TASK3_SAFETY_MARGIN
    )

    print(
        f"\n>>> Token Pattern Extractor: "
        f"{required_tokens}/{TASK3_MAX_CONTEXT}"
    )

    if required_tokens > TASK3_MAX_CONTEXT:
        raise RuntimeError(
            "The Pattern Extractor prompt exceeds the context window: "
            f"{required_tokens}/{TASK3_MAX_CONTEXT}"
        )

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=PATTERN_MAX_OUTPUT_TOKENS
    )

    return (
        response["choices"][0]["message"]["content"]
        .strip()
    )


# ============================================================
# PROMPT TASK 1
# ============================================================

def build_task1_prompt(fake, candidates):

    validation_text = format_task1_candidates(
        candidates
    )

    return f"""
You are an expert Cyber Threat Intelligence analyst.
Your task is to determine whether any candidate validation refutes the specific claim made in the FAKE NEWS.

FAKE NEWS:

ID: {fake["id"]}

Campaign: {fake["campaign"]}

Text:
{fake["text"]}


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
Return ONLY the matching IDs separated by commas (e.g., V06, V07) or NONE.""".strip()


# ============================================================
# PROMPT TASK 2
# ============================================================

def build_task2_prompt(fake, candidates):

    validation_text = ""

    for v in candidates:
        validation_text += (
            f"\nID: {v['id']}\n"
            f"Campaign: {v['campaign']}\n"
            f"Validation: {v['text']}\n"
            f"---\n"
        )

    return f"""
You are an expert Cyber Threat Intelligence (CTI) and Disinformation analyst.

Your task is to select, from the candidate validations, ONLY those that provide evidence that DIRECTLY or INDIRECTLY debunks, refutes, exposes, or analytically undermines a specific claim, source, actor, or documented operation behind the FAKE NEWS below — not evidence that merely shares its general topic or campaign.

FAKE NEWS:
ID: {fake["id"]}
Campaign: {fake["campaign"]}
Text: {fake["text"]}

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
Return ONLY the matching IDs separated by commas (e.g., CT201, AN199, TC200) or NONE.""".strip()


# ============================================================
# MISTRAL
# ============================================================

def ask_mistral(llm, prompt, max_tokens=100):

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=max_tokens
    )

    return (
        response["choices"][0]["message"]["content"]
        .strip()
    )

# ============================================================
# STAGE 1
# ============================================================

def task1(
    fake,
    validations_90_df,
    tfidf_vectorizer,
    tfidf_matrix,
    llm
):

    print("\n" + "=" * 70)
    print("STAGE 1 - ANALYTICAL VALIDATION")
    print("=" * 70)

    candidates = tfidf_retrieval(
        fake_text=fake["text"],
        evidence_df=validations_90_df,
        vectorizer=tfidf_vectorizer,
        evidence_matrix=tfidf_matrix,
        top_k=TOP_K_TASK1
    )

    print("\nTOP 10 TF-IDF")

    for c in candidates:
        print(
            f"{c['id']} -> "
            f"{c['score']:.4f}"
        )

    fake_for_prompt = truncate_fake_for_prompt(
        llm=llm,
        fake=fake,
        candidates=candidates,
        prompt_builder=build_task1_prompt,
        max_context=TASK1_MAX_CONTEXT,
        safety_margin=TASK1_SAFETY_MARGIN
    )

    prompt = build_task1_prompt(
        fake_for_prompt,
        candidates
    )

    output = ask_mistral(
        llm,
        prompt,
        max_tokens=150
    )

    selected_ids = parse_model_output(
        output,
        candidates
    )

    print(
        "\n>>> STAGE 1 OUTPUT:",
        output
    )

    return {
        "stage": 1,
        "source_type": "ANALYTICAL_VALIDATION",
        "retrieval": "TF-IDF",
        "selected_ids": selected_ids,
        "selected": len(selected_ids) > 0,
        "candidates": candidates,
        "raw_output": output
    }


# ============================================================
# STAGE 2
# ============================================================

def task2(
    fake,
    keyword_corpus,
    llm
):

    print("\n" + "=" * 70)
    print("STAGE 2 - FULL CTI EVIDENCE")
    print("=" * 70)

    candidates = keyword_retrieval(
        fake_text=fake["text"],
        keyword_corpus=keyword_corpus,
        top_k=TOP_K_TASK2
    )

    print("\nTOP 10 KEYWORD")

    for c in candidates:
        print(
            f"{c['id']} -> "
            f"{c['score']} keywords"
        )

    fake_for_prompt = truncate_fake_for_prompt(
        llm=llm,
        fake=fake,
        candidates=candidates,
        prompt_builder=build_task2_prompt,
        max_context=TASK2_MAX_CONTEXT,
        safety_margin=TASK2_SAFETY_MARGIN
    )

    prompt = build_task2_prompt(
        fake_for_prompt,
        candidates
    )

    output = ask_mistral(
        llm,
        prompt,
        max_tokens=100
    )

    selected_ids = parse_model_output(
        output,
        candidates
    )

    print(
        "\n>>> STAGE 2 OUTPUT:",
        output
    )

    return {
        "stage": 2,
        "source_type": "CTI_EVIDENCE",
        "retrieval": "KEYWORD",
        "selected_ids": selected_ids,
        "selected": len(selected_ids) > 0,
        "candidates": candidates,
        "raw_output": output
    }


# ============================================================
# STAGE 3
# SAME-CAMPAIGN PATTERN
# ============================================================

def task3(
    fake,
    fake_df,
    e5_model,
    llm
):

    print("\n" + "=" * 70)
    print("STAGE 3 - SAME-CAMPAIGN PATTERN")
    print("=" * 70)

    candidates = campaign_fake_retrieval(
        fake=fake,
        fake_df=fake_df,
        e5_model=e5_model,
        top_k=TOP_K_TASK3
    )

    if not candidates:
        return {
            "stage": 3,
            "source_type": "CAMPAIGN_PATTERN",
            "retrieval": "E5",
            "selected_ids": [],
            "selected": False,
            "candidates": [],
            "raw_output": "NONE"
        }

    print(f"\nTOP {len(candidates)} E5 SAME-CAMPAIGN")

    for c in candidates:
        print(
            f"{c['id']} -> "
            f"{c['score']:.4f}"
        )

    # -----------------------------
    # QUERY FEATURES
    # -----------------------------

    query_feature_result = extract_fake_features(
        llm=llm,
        fake_id=fake["id"],
        campaign=fake["campaign"],
        fake_text=fake["text"]
    )

    query_features = query_feature_result["features"]

    print("\n>>> FEATURE QUERY")
    print(query_features)

    # -----------------------------
    # CANDIDATE FEATURES
    # -----------------------------

    candidate_features = []

    for candidate in candidates:

        feature_result = extract_fake_features(
            llm=llm,
            fake_id=candidate["id"],
            campaign=candidate["campaign"],
            fake_text=candidate["text"]
        )

        candidate_features.append({
            "id": candidate["id"],
            "score": candidate["score"],
            "features": feature_result["features"]
        })

    # -----------------------------
    # PATTERN EXTRACTION
    # -----------------------------

    pattern = extract_pattern(
        llm=llm,
        query_id=fake["id"],
        query_features=query_features,
        candidate_features=candidate_features
    )

    selected = (
        pattern.strip().upper() != "NONE"
    )

    return {
        "stage": 3,
        "source_type": "CAMPAIGN_PATTERN",
        "retrieval": "E5",
        "selected_ids": [],
        "selected": selected,
        "candidates": candidates,
        "query_features": query_features,
        "candidate_features": candidate_features,
        "pattern": pattern,
        "raw_output": pattern
    }


# ============================================================
# CASCADE
# ============================================================

def analyze_fake(
    fake,
    fake_df,
    validations_90_df,
    tfidf_vectorizer,
    tfidf_matrix,
    keyword_corpus,
    e5_model,
    llm
):

    # -----------------------------
    # STAGE 1
    # -----------------------------

    result1 = task1(
        fake=fake,
        validations_90_df=validations_90_df,
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
        llm=llm
    )

    if result1["selected"]:

        print("\nSTOP: evidence selected at STAGE 1")

        return result1


    print(
        "\nTASK 1 = NONE"
        "\nMoving to STAGE 2..."
    )


    # -----------------------------
    # STAGE 2
    # -----------------------------

    result2 = task2(
        fake=fake,
        keyword_corpus=keyword_corpus,
        llm=llm
    )

    if result2["selected"]:

        print("\nSTOP: evidence selected at STAGE 2")

        return result2


    # -----------------------------
    # STAGE 3
    # -----------------------------

    print(
        "\nTASK 1 = NONE"
        "\nTASK 2 = NONE"
        "\nMoving to STAGE 3..."
    )

    result3 = task3(
        fake=fake,
        fake_df=fake_df,
        e5_model=e5_model,
        llm=llm
    )
    return result3

# ============================================================
# EXPLANATION GENERATION
# ============================================================

# ------------------------------------------------------------
# STAGE 1 EXPLANATION PROMPT
# ------------------------------------------------------------

def build_stage1_explanation_prompt(fake, evidence):
    evidence_text = format_evidence(
            evidence
        )
    
    return f"""
You are an expert Disinformation and Cyber Threat Intelligence analyst.

Your task is to explain only the claims or elements of the following fake news
that are actually addressed by the analytical validations provided below.

FAKE NEWS:
ID: {fake["id"]}
Campaign: {fake["campaign"]}
Text: {fake["text"]}

ANALYTICAL VALIDATIONS:
{evidence_text}

INSTRUCTIONS:

Use ONLY the provided validations and explain what they actually establish
about the fake news. Do not use external knowledge or introduce unsupported
facts or conclusions.

Match the strength and scope of the explanation to the evidence:
- DIRECT: only if the validation provides evidence about the same specific
    claim, event, source, actor, or instance described in the fake news.
- PARTIAL: if the validations address only some claims or elements of the
    fake news, explain only those supported parts. Do not draw conclusions
    about other claims or elements that the validations do not address.
- INDIRECT/PATTERN: if the validation concerns a different case or instance,
    even if very similar, treat it only as context. A similar case, historical
    precedent, or recurring pattern does not directly establish that the current
    claim is false.

Use these distinctions internally and do not output the labels.

Omit claims or elements of the fake news that are not addressed by the
provided validations rather than drawing conclusions about them.

Synthesize all relevant validations into one concise, self-contained prose
explanation. Merge overlapping information and avoid repetition.

Do not use lists, headings, evidence IDs, or internal identifiers.

Return only the explanation.
""".strip()


# ------------------------------------------------------------
# STAGE 2 EXPLANATION PROMPT
# ------------------------------------------------------------

def build_stage2_explanation_prompt(fake, evidence):
    evidence_text = format_evidence(
            evidence
        )
    
    return f"""
You are an expert Disinformation and Cyber Threat Intelligence analyst.

Your task is to explain and contextualize the following fake news using
the CTI evidence provided below.

FAKE NEWS:
ID: {fake["id"]}
Campaign: {fake["campaign"]}
Text: {fake["text"]}

CTI EVIDENCE:
{evidence_text}

INSTRUCTIONS:

Use and synthesize ALL the CTI evidence provided below.

The evidence was selected because it may provide direct evidence or specific
CTI context relevant to the fake news. Its selection does NOT mean that it
directly proves, refutes, originates, or belongs to the same operation as the
fake news.

For each piece of evidence, explain only what it explicitly establishes.
Describe its relationship to the fake news without assuming any connection
that is not stated in the evidence.

The fake news may contain multiple independent claims. For each claim or
element discussed in the explanation, first determine whether the provided
evidence actually addresses that same claim or element.

Do not use evidence about one event, actor, or operation to refute a different
claim or event mentioned in the fake news.

When combining multiple pieces of evidence, do not infer a stronger connection
between them than the evidence explicitly supports.

- If the evidence directly establishes that a specific claim is false,
    explain the refutation of that claim only.

- If the evidence provides relevant information about a connected source,
    actor, operation, manipulation, or mechanism but does not establish whether
    a specific claim is true or false, use it only as context and leave that
    claim unresolved.

Do not claim that the entire fake news is false when the provided evidence
supports only a specific part or provides analytical context.

Do not introduce facts, sources, actors, events, or conclusions that are not
supported by the provided evidence.

Do not treat evidence that establishes one event, source, actor, or operation
as proof that another possibility did not occur. Absence of information in
the provided evidence is not evidence that a claim is false.

If the evidence provides relevant context but does not directly establish or
refute the specific claim, state what the evidence establishes without
concluding that the claim is false.

Do not use external knowledge.

A connection between this specific fake news and an actor, source, campaign,
or operation may be stated only if the provided evidence explicitly establishes
that connection. Otherwise, describe the documented actor, source, campaign,
or operation only as context, without suggesting that the fake news may belong
to it or originate from it.

Write a concise, self-contained explanation suitable for a reader who wants
to understand what the available evidence establishes about this fake news.

The evidence IDs are internal metadata used only to identify the retrieved
evidence. Ignore them when writing the explanation. Never copy, mention,
cite, or refer to these IDs in the final answer. Refer only to the factual
content of the evidence.

Do not use lists or headings.

Return only the explanation.
""".strip()


# ------------------------------------------------------------
# STAGE 3 EXPLANATION PROMPT
# ------------------------------------------------------------

def build_stage3_explanation_prompt(fake, pattern):
    return f"""
You are an expert Disinformation and Cyber Threat Intelligence analyst.

Your task is to contextualize the following fake news using a recurring
pattern identified among semantically related fake news from the same
disinformation campaign.

FAKE NEWS:
ID: {fake["id"]}
Campaign: {fake["campaign"]}
Text: {fake["text"]}

CAMPAIGN PATTERN:
{pattern}

INSTRUCTIONS:

Explain how the fake news reflects the recurring campaign pattern described
above.

Use only the information contained in the fake news and the provided pattern.

Focus on the specific characteristics of the fake news that correspond to
the RECURRING ELEMENTS identified in the pattern.

The pattern provides campaign-level analytical context. It does NOT constitute
factual evidence that the specific claim made by the fake news is false.

Do not claim that the fake news has been debunked, disproved, verified as
false, or factually validated solely because it follows the recurring pattern.

Do not infer additional motives, objectives, effects, actors, or facts that
are not explicitly supported by the fake news or the provided pattern.

The supporting fake news indicate only that the pattern recurs within the
campaign. They do NOT constitute factual evidence about the specific claim
in the current fake news.

Supporting fake news IDs are internal metadata. Ignore them when writing the
explanation. Never copy, mention, cite, or refer to these IDs in the final
answer.

Do not use external knowledge.

Write a concise, self-contained explanation of how the fake news reflects
the recurring pattern within the campaign.

Do not use lists or headings.

Return only the explanation.
""".strip()


# ------------------------------------------------------------
# GENERATE EXPLANATION
# ------------------------------------------------------------

def generate_explanation(
    fake,
    result,
    llm,
    max_context=4096,
    max_output_tokens=400,
    safety_margin=200
):

    stage = result["stage"]

    # ========================================================
    # STAGE 1
    # ========================================================

    if stage == 1:

        selected_ids = set(result["selected_ids"])

        selected_evidence = [
            candidate
            for candidate in result["candidates"]
            if candidate["id"] in selected_ids
        ]

        if not selected_evidence:
            raise ValueError(
                "Stage 1 selected but no evidence was found."
            )

        print("\nEvidence used for the Stage 1 explanation:")

        for evidence in selected_evidence:
            print("\nID:", evidence["id"])
            print("Text:", evidence["text"])

        fake_for_generation = truncate_fake_for_generation(
            llm=llm,
            fake=fake,
            prompt_builder=build_stage1_explanation_prompt,
            max_context=max_context,
            max_output_tokens=max_output_tokens,
            safety_margin=safety_margin,
            evidence=selected_evidence
        )

        prompt = build_stage1_explanation_prompt(
            fake=fake_for_generation,
            evidence=selected_evidence
        )

    # ========================================================
    # STAGE 2
    # ========================================================

    elif stage == 2:

        selected_ids = set(result["selected_ids"])

        selected_evidence = [
            candidate
            for candidate in result["candidates"]
            if candidate["id"] in selected_ids
        ]

        if not selected_evidence:
            raise ValueError(
                "Stage 2 selected but no evidence was found."
            )

        print("\nEvidence used for the Stage 2 explanation:")

        for evidence in selected_evidence:
            print("\nID:", evidence["id"])
            print("Text:", evidence["text"])

        fake_for_generation = truncate_fake_for_generation(
            llm=llm,
            fake=fake,
            prompt_builder=build_stage2_explanation_prompt,
            max_context=max_context,
            max_output_tokens=max_output_tokens,
            safety_margin=safety_margin,
            evidence=selected_evidence
        )

        prompt = build_stage2_explanation_prompt(
            fake=fake_for_generation,
            evidence=selected_evidence
        )

    # ========================================================
    # STAGE 3
    # ========================================================

    elif stage == 3:

        pattern = result.get("pattern", "").strip()

        if not pattern or pattern.upper() == "NONE":
            return None

        print("\nPattern used for the Stage 3 explanation:")
        print(pattern)

        fake_for_generation = truncate_fake_for_generation(
            llm=llm,
            fake=fake,
            prompt_builder=build_stage3_explanation_prompt,
            max_context=max_context,
            max_output_tokens=max_output_tokens,
            safety_margin=safety_margin,
            pattern=pattern
        )

        prompt = build_stage3_explanation_prompt(
            fake=fake_for_generation,
            pattern=pattern
        )

    else:

        raise ValueError(
            f"Invalid stage: {stage}"
        )

    # ========================================================
    # TOKEN CHECK
    # ========================================================

    prompt_tokens = len(
        llm.tokenize(
            prompt.encode("utf-8")
        )
    )

    required_tokens = (
        prompt_tokens
        + max_output_tokens
        + safety_margin
    )

    print(
        f"\nExplanation-generation tokens: "
        f"{required_tokens}/{max_context}"
    )

    if required_tokens > max_context:
        raise RuntimeError(
            "The generation prompt exceeds the context window: "
            f"{required_tokens}/{max_context}"
        )

    # ========================================================
    # GENERATION
    # ========================================================

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=max_output_tokens
    )

    explanation = (
        response["choices"][0]["message"]["content"]
        .strip()
    )

    return explanation