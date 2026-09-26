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

MODEL_PATH = SCRIPT_DIR / "prometheus-7b-v2.0.Q4_K_M.gguf"
INPUT_PATH = SCRIPT_DIR / "pipeline_results_1000.xlsx"
OUTPUT_PATH = SCRIPT_DIR / "evaluation_results_1000.xlsx"

for file_path in [MODEL_PATH, INPUT_PATH]:
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")



# ============================================================
# CONFIGURATION
# ============================================================

N_CTX = 4096

MAX_OUTPUT_TOKENS = 400

# Safety margin to prevent context overflow
CONTEXT_MARGIN = 200

TEMPERATURE = 0.0

# Enable to evaluate only a small test subset.
TEST_MODE = False

# 3 cases per stage -> maximum 9 fake-news items
TEST_PER_STAGE = 3


# ============================================================
# GPU SAFETY
# ============================================================

GPU_MAX_TEMP = 80
GPU_RESUME_TEMP = 70
GPU_COOLDOWN_SECONDS = 60

pynvml.nvmlInit()

gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)


def read_gpu_temperature():

    return pynvml.nvmlDeviceGetTemperature(
        gpu_handle,
        pynvml.NVML_TEMPERATURE_GPU
    )


def check_gpu_temperature():

    temperature = read_gpu_temperature()

    print(
        f"GPU temperature: "
        f"{temperature}°C"
    )

    if temperature < GPU_MAX_TEMP:
        return

    print(
        f"\nGPU at {temperature}°C. "
        f"Starting cooldown."
    )

    while temperature > GPU_RESUME_TEMP:

        print(
            f"Waiting {GPU_COOLDOWN_SECONDS} seconds..."
        )

        time.sleep(
            GPU_COOLDOWN_SECONDS
        )

        temperature = read_gpu_temperature()

        print(
            f"GPU temperature: "
            f"{temperature}°C"
        )

    print(
        f"GPU cooled to {temperature}°C "
        f"-> resuming."
    )


# ============================================================
# STAGE SEMANTICS
# ============================================================

STAGE_SEMANTICS = {

    1: """
STAGE 1 - ANALYTICAL VALIDATION

The grounding consists of analytical validation evidence selected
for the current fake news.

Strong conclusions about the fake news are acceptable only when
they are explicitly supported by this evidence.

The evaluator must not use external knowledge.
""".strip(),

    2: """
STAGE 2 - CTI EVIDENCE

The grounding consists of Cyber Threat Intelligence evidence
selected for the current fake news.

This evidence may provide either direct evidence or contextual
information.

The mere presence of this evidence does NOT prove that the specific
fake-news claim is false, nor does it automatically prove that the
fake news was produced by the actor, organization, or operation
described in the evidence.

The evaluator must not use external knowledge.
""".strip(),

    3: """
STAGE 3 - CAMPAIGN PATTERN

The grounding consists of a recurring pattern inferred from other
fake-news items belonging to the same campaign.

The pattern may be used to contextualize the current fake news,
but it does NOT constitute proof that the specific claim is false.

The evaluator must not use external knowledge.
""".strip()
}


# ============================================================
# RUBRICS
# ============================================================

RUBRICS = {

    "FAITHFULNESS": """
[Faithfulness / Grounding]

Evaluate whether the substantive claims made in the generated
explanation are supported by the Relevant Context.

Do NOT use external knowledge.
Do NOT independently determine whether the fake news is objectively
true or false.

Evaluate only whether the explanation is justified by the supplied
grounding context.

Score 1:
The explanation contains major unsupported claims, contradictions,
fabricated information, or conclusions that are not established
by the grounding.

Score 2:
Several important claims go beyond the grounding or substantially
overstate what the available evidence establishes.

Score 3:
The explanation is generally grounded, but contains some unsupported
inferences, overstatements, or insufficiently qualified conclusions.

Score 4:
The explanation is well grounded. Nearly all substantive claims are
supported, with only minor imprecision or weak inference.

Score 5:
All substantive claims are explicitly supported by or cautiously
and logically derived from the supplied grounding. No unsupported
claims or overstatements are present.
""".strip(),

    "RELEVANCE": """
[Relevance]

Evaluate how directly the generated explanation addresses the
specific fake-news claim and provides information useful for
understanding, assessing, or contextualizing that claim.

Score 1:
The explanation is essentially unrelated to the specific fake news.

Score 2:
The explanation mainly discusses the broader campaign or topic and
has only a weak connection to the specific claim.

Score 3:
The explanation addresses the fake news but includes substantial
generic, tangential, or weakly relevant information.

Score 4:
The explanation is clearly focused on the specific fake news, with
only minor unnecessary or generic information.

Score 5:
The explanation is directly and consistently focused on the specific
fake-news claim and provides highly pertinent information for
understanding or assessing it.
""".strip(),

    "COVERAGE": """
[Coverage / Completeness]

Evaluate whether the generated explanation captures the important
information from the supplied grounding that is relevant to
explaining the specific fake news.

Do not reward the inclusion of irrelevant details merely because
they appear in the grounding.

Score 1:
The explanation misses almost all important relevant information
available in the grounding.

Score 2:
The explanation uses only a small portion of the important relevant
grounding and omits major information.

Score 3:
The explanation captures the main point but omits some important
relevant information.

Score 4:
The explanation covers nearly all important relevant information,
with only minor omissions.

Score 5:
The explanation captures all important information from the
grounding that is relevant to explaining the specific fake news,
without unnecessary material.
""".strip()
}


# ============================================================
# SYSTEM PROMPT PROMETHEUS
# ============================================================

SYSTEM_PROMPT = """
You are a fair judge assistant tasked with providing clear,
objective feedback based on specific criteria, ensuring each
assessment reflects the absolute standards set for performance.
""".strip()


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING PIPELINE RESULTS")
print("=" * 70)

df = pd.read_excel(
    INPUT_PATH
)

print(
    f"Total rows: {len(df)}"
)

print(
    f"Evaluable explanations: "
    f"{(df['STATUS'] == 'OK').sum()}"
)

print(
    "\nStage distribution:"
)

print(
    df.loc[
        df["STATUS"] == "OK",
        "STAGE"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# TEST MODE
# ============================================================

if TEST_MODE:

    print("\n" + "=" * 70)
    print("TEST MODE ENABLED")
    print("=" * 70)

    test_parts = []

    for stage in [1, 2, 3]:

        stage_df = df[
            (df["STAGE"] == stage)
            & (df["STATUS"] == "OK")
        ]

        n = min(
            TEST_PER_STAGE,
            len(stage_df)
        )

        if n > 0:

            test_parts.append(
                stage_df.sample(
                    n=n,
                    random_state=42
                )
            )

    df = pd.concat(
        test_parts,
        ignore_index=True
    )

    print(
        f"Fake-news items selected for testing: "
        f"{len(df)}"
    )


# ============================================================
# LOAD PROMETHEUS
# ============================================================

print("\n" + "=" * 70)
print("LOADING PROMETHEUS 2")
print("=" * 70)

llm = Llama(

    model_path=MODEL_PATH,
    n_ctx=N_CTX,
    n_gpu_layers=-1,
    verbose=False
)

print(
    "Prometheus 2 7B Q4_K_M loaded."
)


# ============================================================
# TOKEN UTILITIES
# ============================================================

def count_tokens(text):

    return len(
        llm.tokenize(
            text.encode(
                "utf-8"
            ),
            add_bos=False
        )
    )


def truncate_to_tokens(
    text,
    max_tokens
):

    if max_tokens <= 0:
        return ""

    tokens = llm.tokenize(
        text.encode(
            "utf-8"
        ),
        add_bos=False
    )

    if len(tokens) <= max_tokens:
        return text

    tokens = tokens[
        :max_tokens
    ]

    return llm.detokenize(
        tokens
    ).decode(
        "utf-8",
        errors="ignore"
    )


# ============================================================
# BUILD INSTRUCTION
# ============================================================

def build_instruction(
    fake_text,
    stage
):

    stage = int(stage)

    stage_semantics = (
        STAGE_SEMANTICS[stage]
    )

    return f"""
The response to evaluate is an explanation generated for the
fake-news item below using the supplied Relevant Context.

FAKE NEWS:

{fake_text}

EVALUATION CONDITIONS:

{stage_semantics}

Evaluate the generated explanation only on the basis of the
fake news and the supplied Relevant Context.

Do not use external knowledge.
Do not independently fact-check the fake news.

Do not evaluate whether the Relevant Context was correctly
retrieved or selected, and do not evaluate whether different
or better grounding information could have been retrieved.

The task is only to evaluate the quality of the generated
explanation given the information that was actually available
to the generator.
""".strip()


# ============================================================
# PROMETHEUS RAG PROMPT
# ============================================================

def build_prompt(
    fake_text,
    stage,
    grounding_context,
    explanation,
    rubric
):

    instruction = build_instruction(
        fake_text,
        stage
    )

    user_prompt = f"""
###Task Description:
An instruction, a response to evaluate, a relevant context to support
the evaluation (denoted as Relevant Context), and a score rubric
representing an evaluation criterion are given.

1. Write detailed feedback assessing the quality of the response
strictly according to the given score rubric, not in general.

2. Refer to the Relevant Context when writing the feedback and making
the assessment.

3. After writing the feedback, write a score that is an integer
between 1 and 5.

4. The output format must be:

Feedback: (feedback for the criterion) [RESULT] (integer from 1 to 5)

5. Do not generate any other opening, closing, or explanation.

###The instruction to evaluate:

{instruction}

###Response to evaluate:

{explanation}

###Relevant Context:

{grounding_context}

###Score Rubrics:

{rubric}

###Feedback:
""".strip()

    return user_prompt


# ============================================================
# DYNAMIC FAKE TRUNCATION
# ============================================================

def prepare_prompt(
    fake_text,
    stage,
    grounding_context,
    explanation,
    rubric
):

    # First, try using the complete fake-news text
    prompt = build_prompt(
        fake_text,
        stage,
        grounding_context,
        explanation,
        rubric
    )

    full_text = (
        SYSTEM_PROMPT
        + "\n\n"
        + prompt
    )

    tokens = count_tokens(
        full_text
    )

    available_input_tokens = (
        N_CTX
        - MAX_OUTPUT_TOKENS
        - CONTEXT_MARGIN
    )

    if tokens <= available_input_tokens:

        return (
            prompt,
            False,
            tokens
        )

    # --------------------------------------------------------
    # If it does not fit, calculate the remaining token budget
    # by temporarily removing the fake-news text.
    # Grounding, explanation, and rubric are NOT truncated.
    # --------------------------------------------------------

    empty_fake_prompt = build_prompt(
        "",
        stage,
        grounding_context,
        explanation,
        rubric
    )

    base_text = (
        SYSTEM_PROMPT
        + "\n\n"
        + empty_fake_prompt
    )

    base_tokens = count_tokens(
        base_text
    )

    fake_budget = (
        available_input_tokens
        - base_tokens
    )

    if fake_budget <= 0:

        raise ValueError(
            "Grounding + explanation + rubric "
            "exceed the available context window."
        )

    truncated_fake = truncate_to_tokens(
        fake_text,
        fake_budget
    )

    prompt = build_prompt(
        truncated_fake,
        stage,
        grounding_context,
        explanation,
        rubric
    )

    final_tokens = count_tokens(
        SYSTEM_PROMPT
        + "\n\n"
        + prompt
    )

    return (
        prompt,
        True,
        final_tokens
    )


# ============================================================
# PARSE PROMETHEUS OUTPUT
# ============================================================

def parse_prometheus_output(
    output
):

    match = re.search(
        r"\[RESULT\]\s*([1-5])",
        output,
        flags=re.IGNORECASE
    )

    if not match:

        raise ValueError(
            f"Score [RESULT] not found "
            f"in output:\n{output}"
        )

    score = int(
        match.group(1)
    )

    feedback = output[
        :match.start()
    ].strip()

    feedback = re.sub(
        r"^Feedback:\s*",
        "",
        feedback,
        flags=re.IGNORECASE
    ).strip()

    return (
        score,
        feedback
    )


# ============================================================
# EVALUATE ONE CRITERION
# ============================================================

def evaluate_criterion(
    fake_text,
    stage,
    grounding_context,
    explanation,
    criterion
):

    rubric = RUBRICS[
        criterion
    ]

    prompt, truncated, input_tokens = (
        prepare_prompt(
            fake_text,
            stage,
            grounding_context,
            explanation,
            rubric
        )
    )

    print(
        f"{criterion}: "
        f"{input_tokens} input tokens"
        + (
            " [FAKE TRUNCATED]"
            if truncated
            else ""
        )
    )

    response = llm.create_chat_completion(

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=TEMPERATURE,

        max_tokens=MAX_OUTPUT_TOKENS
    )

    output = (
        response["choices"][0]
        ["message"]["content"]
        .strip()
    )

    score, feedback = (
        parse_prometheus_output(
            output
        )
    )

    return {
        "score": score,
        "feedback": feedback,
        "raw_output": output,
        "fake_truncated": truncated,
        "input_tokens": input_tokens
    }


# ============================================================
# CHECKPOINT / RESUME
# ============================================================

if OUTPUT_PATH.exists():

    print("\n" + "=" * 70)
    print("JUDGE CHECKPOINT FOUND")
    print("=" * 70)

    previous_df = pd.read_excel(
        OUTPUT_PATH,
        sheet_name="RESULTS"
    )

    completed_ids = set(
        previous_df.loc[
            previous_df["JUDGE_STATUS"]
            == "OK",
            "ID_FAKE"
        ]
        .astype(str)
        .str.strip()
    )

    results = previous_df.to_dict(
        orient="records"
    )

    print(
        f"Already evaluated: "
        f"{len(completed_ids)}"
    )

else:

    completed_ids = set()
    results = []


# ============================================================
# JUDGE LOOP
# ============================================================

total = len(df)

for position, row in df.iterrows():

    fake_id = str(
        row["ID_FAKE"]
    ).strip()

    pipeline_status = str(
        row["STATUS"]
    ).strip()

    # --------------------------------------------------------
    # PIPELINE ERROR -> keep the reference, do not run Judge
    # --------------------------------------------------------

    if pipeline_status != "OK":

        print(
            f"\n{fake_id} cannot be evaluated "
            f"(Pipeline STATUS={pipeline_status}) "
            f"-> SKIP JUDGE"
        )

        record = {
            "SAMPLE_INDEX": row["SAMPLE_INDEX"],
            "ID_FAKE": fake_id,
            "CAMPAIGN": row["CAMPAIGN"],
            "STAGE": row["STAGE"],
            "SOURCE_TYPE": row["SOURCE_TYPE"],

            "FAITHFULNESS": "",
            "FAITHFULNESS_FEEDBACK": "",

            "RELEVANCE": "",
            "RELEVANCE_FEEDBACK": "",

            "COVERAGE": "",
            "COVERAGE_FEEDBACK": "",

            "FAKE_TRUNCATED": "",
            "MAX_INPUT_TOKENS": "",

            "JUDGE_STATUS": "",
            "JUDGE_ERROR": ""
        }

        results = [
            r for r in results
            if str(r.get("ID_FAKE", "")).strip() != fake_id
        ]

        results.append(
            record
        )

        results_df = pd.DataFrame(
            results
        )

        results_df = results_df.sort_values(
            "SAMPLE_INDEX"
        )

        results_df.to_excel(
            OUTPUT_PATH,
            sheet_name="RESULTS",
            index=False
        )

        continue

    # --------------------------------------------------------
    # Already evaluated by Judge
    # --------------------------------------------------------

    if fake_id in completed_ids:

        print(
            f"\n{fake_id} already evaluated "
            f"-> SKIP"
        )

        continue

    print("\n" + "=" * 70)

    print(
        f"JUDGE {len(completed_ids) + 1}/{total} "
        f"- FAKE {fake_id}"
    )

    print(
        f"STAGE {int(row['STAGE'])}"
    )

    print("=" * 70)

    fake_text = str(
        row["FAKE_TEXT"]
        if pd.notna(row["FAKE_TEXT"])
        else ""
    )

    grounding_context = str(
        row["GROUNDING_CONTEXT"]
        if pd.notna(row["GROUNDING_CONTEXT"])
        else ""
    )

    explanation = str(
        row["EXPLANATION"]
        if pd.notna(row["EXPLANATION"])
        else ""
    )

    try:

        evaluations = {}

        # ----------------------------------------------------
        # 3 independent evaluations
        # ----------------------------------------------------

        for criterion in [
            "FAITHFULNESS",
            "RELEVANCE",
            "COVERAGE"
        ]:

            check_gpu_temperature()

            print(
                f"\nEvaluating "
                f"{criterion}..."
            )

            evaluations[
                criterion
            ] = evaluate_criterion(

                fake_text=fake_text,

                stage=row["STAGE"],

                grounding_context=grounding_context,

                explanation=explanation,

                criterion=criterion
            )

            print(
                f"Score: "
                f"{evaluations[criterion]['score']}/5"
            )

            print(
                "Feedback:"
            )

            print(
                evaluations[
                    criterion
                ]["feedback"]
            )

        # ----------------------------------------------------
        # SAVE RECORD
        # ----------------------------------------------------

        record = {

            "SAMPLE_INDEX":
                row["SAMPLE_INDEX"],

            "ID_FAKE":
                fake_id,

            "CAMPAIGN":
                row["CAMPAIGN"],

            "STAGE":
                int(row["STAGE"]),

            "SOURCE_TYPE":
                row["SOURCE_TYPE"],

            "FAITHFULNESS":
                evaluations[
                    "FAITHFULNESS"
                ]["score"],

            "FAITHFULNESS_FEEDBACK":
                evaluations[
                    "FAITHFULNESS"
                ]["feedback"],

            "RELEVANCE":
                evaluations[
                    "RELEVANCE"
                ]["score"],

            "RELEVANCE_FEEDBACK":
                evaluations[
                    "RELEVANCE"
                ]["feedback"],

            "COVERAGE":
                evaluations[
                    "COVERAGE"
                ]["score"],

            "COVERAGE_FEEDBACK":
                evaluations[
                    "COVERAGE"
                ]["feedback"],

            "FAKE_TRUNCATED":
                any(
                    evaluations[c][
                        "fake_truncated"
                    ]
                    for c in evaluations
                ),

            "MAX_INPUT_TOKENS":
                max(
                    evaluations[c][
                        "input_tokens"
                    ]
                    for c in evaluations
                ),

            "JUDGE_STATUS":
                "OK",

            "JUDGE_ERROR":
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

    except Exception as e:

        print(
            f"\nJUDGE ERROR FOR "
            f"{fake_id}: {e}"
        )

        record = {

            "SAMPLE_INDEX":
                row["SAMPLE_INDEX"],

            "ID_FAKE":
                fake_id,

            "CAMPAIGN":
                row["CAMPAIGN"],

            "STAGE":
                row["STAGE"],

            "SOURCE_TYPE":
                row["SOURCE_TYPE"],

            "FAITHFULNESS":
                "",

            "FAITHFULNESS_FEEDBACK":
                "",

            "RELEVANCE":
                "",

            "RELEVANCE_FEEDBACK":
                "",

            "COVERAGE":
                "",

            "COVERAGE_FEEDBACK":
                "",

            "FAKE_TRUNCATED":
                "",

            "MAX_INPUT_TOKENS":
                "",

            "JUDGE_STATUS":
                "ERROR",

            "JUDGE_ERROR":
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

    results_df = (
        results_df
        .sort_values(
            "SAMPLE_INDEX"
        )
    )

    results_df.to_excel(
        OUTPUT_PATH,
        sheet_name="RESULTS",
        index=False
    )

    print(
        f"\nJudge checkpoint saved -> "
        f"{OUTPUT_PATH}"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("JUDGE COMPLETED")
print("=" * 70)

results_df = (
    pd.DataFrame(results)
    .sort_values("SAMPLE_INDEX")
    .reset_index(drop=True)
)

ok_judge = results_df[
    results_df["JUDGE_STATUS"]
    == "OK"
]

error_judge = results_df[
    results_df["JUDGE_STATUS"]
    == "ERROR"
]

print(
    f"Completed evaluations: "
    f"{len(ok_judge)}"
)

print(
    f"Judge errors: "
    f"{len(error_judge)}"
)

if len(ok_judge) > 0:

    print(
        "\nOVERALL MEANS:"
    )

    print(
        ok_judge[
            [
                "FAITHFULNESS",
                "RELEVANCE",
                "COVERAGE"
            ]
        ].mean()
    )

    print(
        "\nMEANS BY STAGE:"
    )

    print(
        ok_judge.groupby(
            "STAGE"
        )[
            [
                "FAITHFULNESS",
                "RELEVANCE",
                "COVERAGE"
            ]
        ].mean()
    )

# ============================================================
# SAVE FINAL SUMMARY
# ============================================================

summary_rows = [
    {
        "SECTION": "GENERAL",
        "GROUP": "Completed evaluations",
        "FAITHFULNESS": len(ok_judge),
        "RELEVANCE": "",
        "COVERAGE": ""
    },
    {
        "SECTION": "GENERAL",
        "GROUP": "Judge errors",
        "FAITHFULNESS": len(error_judge),
        "RELEVANCE": "",
        "COVERAGE": ""
    },
    {
        "SECTION": "OVERALL MEAN",
        "GROUP": "ALL",
        "FAITHFULNESS": ok_judge["FAITHFULNESS"].mean(),
        "RELEVANCE": ok_judge["RELEVANCE"].mean(),
        "COVERAGE": ok_judge["COVERAGE"].mean()
    }
]

for stage, stage_df in ok_judge.groupby("STAGE"):

    summary_rows.append(
        {
            "SECTION": "MEAN BY STAGE",
            "GROUP": f"STAGE {int(stage)}",
            "FAITHFULNESS": stage_df["FAITHFULNESS"].mean(),
            "RELEVANCE": stage_df["RELEVANCE"].mean(),
            "COVERAGE": stage_df["COVERAGE"].mean()
        }
    )

summary_df = pd.DataFrame(summary_rows)

with pd.ExcelWriter(
    OUTPUT_PATH,
    engine="openpyxl"
) as writer:

    results_df.to_excel(
        writer,
        sheet_name="RESULTS",
        index=False
    )

    summary_df.to_excel(
        writer,
        sheet_name="SUMMARY",
        index=False
    )

print(
    f"\nResults saved to:\n"
    f"{OUTPUT_PATH}"
)