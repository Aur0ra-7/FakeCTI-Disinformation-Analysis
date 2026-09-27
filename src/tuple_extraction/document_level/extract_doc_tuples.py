import re
import time
from pathlib import Path

import pandas as pd
import pynvml
from llama_cpp import Llama


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_PATH = SCRIPT_DIR / "Llama-3-8B-Instruct-v0.10.Q5_K_M.gguf"
DOCUMENTS_DIR = SCRIPT_DIR / "documents"
OUTPUT_FILE = SCRIPT_DIR / "Extracted_Doc_Tuples.xlsx"


# ---------------------------------------------------------------------------
# GPU monitoring
# ---------------------------------------------------------------------------

pynvml.nvmlInit()


def get_gpu_temp():
    """Return the current GPU temperature in Celsius."""

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetTemperature(
            handle,
            pynvml.NVML_TEMPERATURE_GPU,
        )
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def model_run(llm, user_input):
    """Run the model on the provided prompt."""

    stop_sequence = [
        "##END LIST##",
        "#END LIST#",
        "END LIST",
    ]

    output = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract subject-verb-object tuples from text "
                    "accurately and concisely."
                ),
            },
            {
                "role": "user",
                "content": user_input,
            },
        ],
        temperature=0.2,
        top_p=0.95,
        stop=stop_sequence,
        max_tokens=512,
    )

    return output["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Tuple extraction and cleaning
# ---------------------------------------------------------------------------

def extract_all_tuples(text):
    """Extract SVO tuples from the model output."""

    pattern = r"([^\n]+? - [^\n]+? - [^\n]+?)\n"
    return re.findall(pattern, text)


def clean_tuples(tuples):
    """Remove malformed, repetitive, or low-quality tuples."""

    cleaned_set = set()
    final_tuples = []

    invalid_starts = (
        "is -",
        "are -",
        "was -",
        "were -",
        "it -",
        "there -",
        "this -",
        "that -",
        "- -",
    )

    for tuple_text in tuples:

        tuple_text = re.sub(
            r"\s+",
            " ",
            tuple_text.strip(),
        )

        # Require exactly two separators: Subject - Verb - Object
        if tuple_text.count(" - ") != 2:
            continue

        tuple_lower = tuple_text.lower()

        # Remove tuples starting with empty or impersonal subjects
        if any(
            tuple_lower.startswith(prefix)
            for prefix in invalid_starts
        ):
            continue

        parts = tuple_text.split(" - ")

        if len(parts) != 3:
            continue

        subject = parts[0].strip()
        verb = parts[1].strip()
        obj = parts[2].strip()

        # Remove incomplete fragments
        if (
            len(subject) < 2
            or len(verb) < 2
            or len(obj) < 2
        ):
            continue

        # Case-insensitive duplicate removal
        if tuple_lower not in cleaned_set:
            cleaned_set.add(tuple_lower)
            final_tuples.append(tuple_text)

    return final_tuples


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def create_full_prompt(text):
    """Create the tuple-extraction prompt."""

    prompt = f"""Instructions:
        1. Focus only on the main factual actions or events.
        2. Each tuple must contain:
        - A clear subject (the entity performing the action),
        - A precise verb (the action or relation),
        - A meaningful object (the entity affected or involved).
        3. Include factual statements even if they appear within direct quotes, as long as the speaker is identifiable or can be inferred from context.
        - Use the speaker as the subject of the tuple.
        - Example: if the text is 'John said, "We support the initiative."', extract: John - said - we support the initiative.
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

        Now, extract tuples from this text: {text}"""

    return prompt


# ---------------------------------------------------------------------------
# Document chunking
# ---------------------------------------------------------------------------

def split_into_chunks(text, max_characters=3000):
    """
    Split a document into chunks, preferably at sentence boundaries
    or line breaks.
    """

    chunks = []

    while len(text) > max_characters:

        split_point = max(
            text.rfind(". ", 0, max_characters),
            text.rfind("\n", 0, max_characters),
        )

        if (
            split_point == -1
            or split_point < max_characters // 2
        ):
            split_point = max_characters
        else:
            split_point += 1

        chunks.append(
            text[:split_point].strip()
        )

        text = text[split_point:].strip()

    if text:
        chunks.append(text)

    return chunks


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def main():

    print("=== DOCUMENT-LEVEL TUPLE EXTRACTION ===")
    print(f"Documents directory: {DOCUMENTS_DIR}")
    print(f"Output file: {OUTPUT_FILE}")

    if not DOCUMENTS_DIR.exists():
        print(
            f"ERROR: Documents directory not found: "
            f"{DOCUMENTS_DIR}"
        )
        return

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------

    print("Loading model on GPU...")

    llm = Llama(
        model_path=str(MODEL_PATH),
        n_gpu_layers=-1,
        n_ctx=4096,
        verbose=False,
    )

    print("Model loaded successfully.")

    # -----------------------------------------------------------------------
    # Resume from previous output
    # -----------------------------------------------------------------------

    if OUTPUT_FILE.exists():

        completed_df = pd.read_excel(OUTPUT_FILE)
        results = completed_df.to_dict("records")

        completed_documents = set(
            zip(
                completed_df["CAMPAGNA"],
                completed_df["DOCUMENTO"],
            )
        )

        print(
            f"Resuming previous extraction: "
            f"{len(completed_documents)} documents already processed."
        )

    else:
        completed_documents = set()
        results = []

        print("Starting a new extraction.")

    # -----------------------------------------------------------------------
    # Scan campaign directories
    # -----------------------------------------------------------------------

    campaign_dirs = [
        directory
        for directory in DOCUMENTS_DIR.iterdir()
        if directory.is_dir()
    ]

    print(
        f"Found {len(campaign_dirs)} campaign directories: "
        f"{[directory.name for directory in campaign_dirs]}"
    )

    processed_files = 0

    for campaign_dir in campaign_dirs:

        campaign_name = campaign_dir.name

        document_files = [
            file
            for file in campaign_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() == ".txt"
        ]

        print(
            f"\n[CAMPAIGN] {campaign_name} "
            f"({len(document_files)} .txt files found)"
        )

        for document_file in document_files:

            document_name = document_file.name

            if (
                campaign_name,
                document_name,
            ) in completed_documents:
                print(
                    f"   [SKIP] Already processed: "
                    f"{document_name}"
                )
                continue

            print(
                f"   [PROCESSING] Reading document: "
                f"{document_name}"
            )

            try:
                text = document_file.read_text(
                    encoding="utf-8"
                ).strip()

            except Exception as exc:
                print(
                    f"   [ERROR] Unable to read "
                    f"{document_name}: {exc}"
                )
                continue

            if not text:
                print(
                    f"   [WARNING] Empty document: "
                    f"{document_name}. Skipping."
                )
                continue

            # ---------------------------------------------------------------
            # Split document into chunks
            # ---------------------------------------------------------------

            chunks = split_into_chunks(
                text,
                max_characters=3000,
            )

            print(
                f"   -> Document split into "
                f"{len(chunks)} chunks."
            )

            document_tuples = []

            # ---------------------------------------------------------------
            # Extract tuples from each chunk
            # ---------------------------------------------------------------

            for index, chunk in enumerate(chunks):

                temperature = get_gpu_temp()

                if temperature > 80:
                    print(
                        f"   [WARNING] High GPU temperature "
                        f"({temperature}°C). Pausing for 60 seconds..."
                    )
                    time.sleep(60)

                try:
                    user_input = create_full_prompt(chunk)

                    response = model_run(
                        llm,
                        user_input,
                    )

                    tuple_list = extract_all_tuples(
                        response + "\n"
                    )

                    cleaned_tuples = clean_tuples(
                        tuple_list
                    )

                    if cleaned_tuples:
                        document_tuples.extend(
                            cleaned_tuples
                        )

                except Exception as exc:
                    print(
                        f"   [ERROR - CHUNK {index + 1}] "
                        f"Inference failed: {exc}"
                    )

            # Remove duplicate tuples within the same document
            document_tuples = list(
                dict.fromkeys(document_tuples)
            )

            if document_tuples:

                print(
                    f"   -> Extracted "
                    f"{len(document_tuples)} unique tuples."
                )

                for tuple_text in document_tuples:
                    results.append(
                        {
                            "CAMPAGNA": campaign_name,
                            "DOCUMENTO": document_name,
                            "TUPLA": tuple_text,
                        }
                    )

            else:

                print(
                    "   -> No valid tuples extracted "
                    "from this document."
                )

                results.append(
                    {
                        "CAMPAGNA": campaign_name,
                        "DOCUMENTO": document_name,
                        "TUPLA": "Nessuna tupla valida estratta",
                    }
                )

            processed_files += 1

            # ---------------------------------------------------------------
            # Incremental checkpoint
            # ---------------------------------------------------------------

            output_df = pd.DataFrame(results)
            output_df.to_excel(
                OUTPUT_FILE,
                index=False,
            )

            print(
                f"   -> Checkpoint saved. "
                f"Total rows: {len(results)}"
            )

    pynvml.nvmlShutdown()

    print("\n==========================================")
    print(
        f"Completed. Processed {processed_files} "
        f"documents in this session."
    )
    print(f"Output file: {OUTPUT_FILE}")
    print("==========================================")


if __name__ == "__main__":
    main()