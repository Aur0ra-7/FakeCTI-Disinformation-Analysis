# Tuple Extraction

This module contains the experiments for extracting structured Subject–Verb–Object (SVO) tuples from Cyber Threat Intelligence (CTI) information related to disinformation campaigns.

Two complementary extraction settings were investigated:

1. **Evidence-level extraction**, used as the main strategy for the structured representation of CTI knowledge.
2. **Document-level extraction**, investigated as an exploratory alternative in which tuples are extracted directly from the original source documents.

Both experiments use a locally executed Llama 3 8B model through `llama.cpp`.

---

## Repository Structure

```text
data/
├── categorie_manuali_doppie.csv
└── tuple_extraction/
    └── document_level/
        └── Source_12_Documents.xlsx

src/
└── tuple_extraction/
    ├── README.md
    │
    ├── evidence_level/
    │   ├── extract_tuples.py
    │   ├── Extracted_Tuples.xlsx
    │   └── Manual_Evaluation_Sample.xlsx
    │
    └── document_level/
        ├── extract_documents.py
        ├── extract_doc_tuples.py
        ├── documents/
        │   └── <campaign>/
        │       └── *.txt
        ├── Extracted_Doc_Tuples.xlsx
        └── Document_Tuples_Manual_Evaluation.xlsx
```

The GGUF model files are not included in the repository and must be downloaded separately.

---

## 1. Evidence-Level Tuple Extraction

The first experiment extracts SVO tuples directly from the CTI evidences contained in:

```text
data/categorie_manuali_doppie.csv
```

Each evidence is associated with a disinformation campaign and with its manually assigned CTI category or categories.

### Extraction

The extraction is performed by:

```text
src/tuple_extraction/evidence_level/extract_tuples.py
```

For each evidence, the script:

1. reads the evidence text and its metadata;
2. removes citation information from the text before inference;
3. provides the cleaned evidence to the LLM;
4. asks the model to extract factual relationships in the form:

```text
Subject - Verb - Object
```

5. parses the model output and stores the extracted tuples together with the corresponding campaign, evidence and CTI categories.

The experiment uses the following main inference parameters:

```text
temperature = 0.3
top_p = 0.95
context size = 2048
GPU layers = -1
```

The resulting tuples are stored in:

```text
evidence_level/Extracted_Tuples.xlsx
```

The extraction produced **2,352 tuples**.

### Manual Evaluation

A subset of 94 evidences, covering all 43 disinformation campaigns, was selected for manual evaluation. For each campaign, one third of the available evidences was selected, rounding down to the nearest integer. All tuples extracted from the selected evidences were then included in the evaluation, resulting in 623 manually evaluated tuples.

The annotations are available in:

```text
evidence_level/Manual_Evaluation_Sample.xlsx
```

The following labels are used:

- `OK` — semantically correct tuple.
- `SPEZZ` — correct but incomplete or slightly imprecise tuple.
- `RIP` — repeated or redundant tuple.
- `NO` — semantically incorrect tuple.

The annotation file is provided to preserve the manual evaluation used in the experiment.

---

## 2. Document-Level Tuple Extraction

The second experiment evaluates tuple extraction from longer source documents rather than from individual CTI evidences.

A set of **12 source documents** was selected for this experiment. The documents belong to three disinformation campaigns: *Cannabis Cure for Cancer* (6 documents), *Murder of Marilyn Monroe* (4 documents), and *Greta Thunberg Banned from Twitter* (2 documents). The source list is available in:

```text
data/tuple_extraction/document_level/Source_12_Documents.xlsx
```

The file contains the source description, URL and associated disinformation campaign.

The document-level workflow consists of two stages.

### Stage 1 — Document Acquisition

Source documents are retrieved using:

```text
src/tuple_extraction/document_level/extract_documents.py
```

The script first attempts to retrieve each URL using `trafilatura`. If the initial download fails, a second attempt is performed using `requests`.

For standard web sources, the complete text extracted by `trafilatura` is retained.

For Wikipedia sources, an additional filtering step is applied. Keywords are derived from the corresponding campaign name after removing common English stopwords. Lines containing at least one campaign keyword are retained together with one preceding and one following line of context.

The processed documents are stored as UTF-8 text files under:

```text
document_level/documents/<campaign>/
```

Each file also contains the campaign, source and original URL as metadata.

#### Manually Retrieved Documents

Most documents were acquired automatically by `extract_documents.py`.

When automatic retrieval failed, the corresponding source content was retrieved manually and added to the `documents/` directory. These documents were subsequently processed by the **same automatic tuple extraction pipeline** as all other documents.

Rows associated with these manually retrieved source documents are highlighted in yellow in:

```text
Document_Tuples_Manual_Evaluation.xlsx
```

The highlighting therefore refers only to the **document acquisition step** and does not indicate manually generated tuples.

### Stage 2 — Tuple Extraction

Tuple extraction from the collected documents is performed by:

```text
src/tuple_extraction/document_level/extract_doc_tuples.py
```

Because full documents may exceed the available model context, each document is divided into chunks of at most **3,000 characters**.

Whenever possible, chunk boundaries are placed at sentence endings or line breaks to avoid splitting sentences unnecessarily.

Each chunk is independently provided to the LLM, which extracts relationships using the format:

```text
Subject - Verb - Object
```

The extraction uses:

```text
temperature = 0.2
top_p = 0.95
max_tokens = 512
context size = 4096
GPU layers = -1
```

The extracted tuples are automatically filtered to remove malformed outputs and tuples that do not satisfy the expected structural constraints. The cleaning step:

- requires exactly three SVO components;
- removes tuples with empty or impersonal subjects;
- removes incomplete components;
- performs case-insensitive duplicate removal.

Duplicates occurring across different chunks of the same document are also removed before saving the final document-level tuples.

The script supports incremental checkpointing: if an output file already exists, previously processed campaign-document pairs are detected and skipped.

The extracted tuples are stored in:

```text
document_level/Extracted_Doc_Tuples.xlsx
```

with the following fields:

```text
CAMPAGNA
DOCUMENTO
TUPLA
```

---

## Document-Level Manual Evaluation

The document-level experiment produced **1,873 tuples**: 1,480 from *Cannabis Cure for Cancer*, 359 from *Murder of Marilyn Monroe*, and 34 from *Greta Thunberg Banned from Twitter*.

All extracted tuples were manually evaluated. The complete annotations are available in:

```text
document_level/Document_Tuples_Manual_Evaluation.xlsx
```

The manual evaluation uses the following labels:

- `OK` — semantically correct tuple.
- `SPEZZ` — correct but incomplete or slightly imprecise tuple.
- `RIP` — repeated or redundant tuple.
- `NO` — semantically incorrect tuple.
- `NOISE` — semantically correct with respect to the source document, but not informative for the CTI analysis of interest, for example because the tuple is derived from questions, references, or quotations occurring in the document.

The `NOISE` class is specific to the document-level analysis and distinguishes extraction errors from valid information that is outside the relevant CTI knowledge scope.

---

## Model

Both tuple extraction experiments use:

**Llama-3-8B-Instruct-v0.10.Q5_K_M**

The model is executed locally through `llama-cpp-python`.

Model:

https://huggingface.co/MaziyarPanahi/Llama-3-8B-Instruct-v0.10-GGUF

Download the corresponding GGUF file and place it in the directory of the extraction script before execution.

Expected filename:

```text
Llama-3-8B-Instruct-v0.10.Q5_K_M.gguf
```

Model files are excluded from the repository through `.gitignore`.

---

## Requirements

The main Python dependencies are:

```text
pandas
openpyxl
llama-cpp-python
pynvml
trafilatura
requests
```

A CUDA-compatible environment is recommended for GPU inference with `llama.cpp`.

---

## Execution

### Evidence-Level Extraction

From the repository root:

```bash
python src/tuple_extraction/evidence_level/extract_tuples.py
```

The script reads:

```text
data/categorie_manuali_doppie.csv
```

and generates:

```text
src/tuple_extraction/evidence_level/Extracted_Tuples.xlsx
```

### Document Acquisition

Retrieve and preprocess the selected source documents with:

```bash
python src/tuple_extraction/document_level/extract_documents.py
```

The script reads:

```text
data/tuple_extraction/document_level/Source_12_Documents.xlsx
```

and stores the processed documents under:

```text
src/tuple_extraction/document_level/documents/
```

### Document-Level Tuple Extraction

After the document texts are available, run:

```bash
python src/tuple_extraction/document_level/extract_doc_tuples.py
```

The script processes all `.txt` files contained in the campaign directories under `documents/` and generates:

```text
src/tuple_extraction/document_level/Extracted_Doc_Tuples.xlsx
```

The `documents/` directory included in the repository contains the document texts used for the reported experiment.