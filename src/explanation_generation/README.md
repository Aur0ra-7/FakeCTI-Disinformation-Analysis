# Explanation Generation

This directory contains the scripts and files used for the explanation-generation experiment on the FakeCTI dataset.

The pipeline processes a fixed sample of 1,000 fake-news items using a three-stage grounding strategy. Depending on the available grounding, the generated explanation is based on analytical validations, CTI evidence, or recurring patterns identified within the same disinformation campaign.

## Directory Structure

```text
src/explanation_generation/
├── README.md
├── create_sample.py
├── pipeline.py
├── run_pipeline.py
├── evaluate_explanations.py
├── sample_1000.xlsx
├── pipeline_results_1000.xlsx
└── evaluation_results_1000.xlsx
```

## Files

### `create_sample.py`

Handles the experimental sample used by the pipeline.

By default, the script loads and validates the official `sample_1000.xlsx` included in this directory. The sample contains 1,000 fake-news items selected from the complete FakeCTI dataset.

The script can also generate a new sample from `data/FakeCTI.csv` by setting:

```python
REFERENCE_SAMPLE = None
```

The sampling procedure uses disproportionate stratified random sampling across the disinformation campaigns, with a minimum campaign quota followed by proportional allocation of the remaining observations.

### `sample_1000.xlsx`

Contains the official 1,000-instance sample used for the experiment.

The workbook includes:

- `SAMPLE` — fake-news items included in the experimental sample;
- `DISTRIBUTION` — comparison between campaign distributions in FakeCTI and in the sample;
- `ALLOCATION` — number of dataset and sampled instances for each campaign;
- `METADATA` — sampling configuration and general information about the sample.

The pipeline uses the `SAMPLE` sheet directly.

### `pipeline.py`

Contains the core functions of the explanation-generation pipeline.

The pipeline is organized into three stages:

1. **Stage 1 — Analytical Validation**  
   TF-IDF retrieves the top 10 candidates from the 90 analytical validations. Mistral-Nemo selects the validations that can provide grounding for the current fake-news item.

2. **Stage 2 — CTI Evidence**  
   If Stage 1 does not select any validation, keyword retrieval retrieves the top 10 candidates from the complete set of 319 CTI evidences. Mistral-Nemo selects the relevant evidence.

3. **Stage 3 — Campaign Pattern**  
   If no evidence is selected in the previous stages, E5 retrieves similar fake-news items belonging to the same campaign. Their characteristics are analyzed to identify a recurring campaign pattern that can be used as contextual grounding.

The file also contains the prompt construction, candidate selection, campaign-pattern extraction, context management, and final explanation-generation functions.

### `run_pipeline.py`

Runs the complete explanation-generation experiment on `sample_1000.xlsx`.

The script:

- loads `FakeCTI.csv`;
- loads the 90 analytical validations;
- loads the 319 CTI evidences;
- loads the official 1,000-instance sample;
- initializes the TF-IDF and keyword retrieval indexes;
- loads the E5 embedding model;
- loads Mistral-Nemo;
- executes the three-stage pipeline for each fake-news item;
- generates the final explanation;
- reconstructs and stores the grounding context used for generation;
- saves a checkpoint after each processed instance;
- supports resuming an interrupted execution.

The resulting file is:

```text
pipeline_results_1000.xlsx
```

### `pipeline_results_1000.xlsx`

Contains the output produced by `run_pipeline.py`.

For each fake-news item, the file stores information including:

- sample index;
- fake-news ID;
- campaign;
- fake-news text;
- stage used for explanation generation;
- grounding source type;
- selected evidence IDs, when applicable;
- grounding context;
- campaign pattern, when applicable;
- generated explanation;
- execution status;
- error information.

### `evaluate_explanations.py`

Evaluates the explanations contained in `pipeline_results_1000.xlsx` using **Prometheus 2 7B** as an LLM-as-a-Judge.

Each successfully generated explanation is evaluated independently according to three criteria:

- **Faithfulness** — consistency of the explanation with the supplied grounding;
- **Relevance** — relevance of the explanation to the specific fake-news item;
- **Coverage** — coverage of the relevant information available in the grounding.

Each criterion is evaluated using absolute grading on a scale from 1 to 5.

The evaluator receives the fake-news text, the generated explanation, the grounding context used by the generator, the semantics of the corresponding pipeline stage, and the criterion-specific rubric.

The evaluation does not perform a new retrieval step and does not assess whether different evidence should have been retrieved.

The resulting file is:

```text
evaluation_results_1000.xlsx
```

### `evaluation_results_1000.xlsx`

Contains the output produced by `evaluate_explanations.py`.

The workbook includes:

- `RESULTS` — per-instance Prometheus evaluations, scores, feedback, and execution information;
- `SUMMARY` — aggregate evaluation statistics.

## Input Data

The scripts use the following files from the repository:

```text
data/
├── FakeCTI.csv
└── coverage/
    ├── Analytical_Validations_90.xlsx
    └── CTI_Evidences_319.xlsx
```

The analytical validations and CTI evidences are shared with the Coverage experiment and are therefore not duplicated in this directory.

## Models

### Mistral-Nemo

The explanation pipeline uses:

```text
Mistral-Nemo-Instruct-2407-Q4_K_S.gguf
```

Mistral-Nemo is used for evidence selection, feature and campaign-pattern extraction, and explanation generation.

Download:

https://huggingface.co/mradermacher/Mistral-Nemo-Instruct-2407-GGUF

Place the downloaded file in:

```text
src/explanation_generation/
```

so that the final path is:

```text
src/explanation_generation/Mistral-Nemo-Instruct-2407-Q4_K_S.gguf
```

### E5

Stage 3 uses:

```text
intfloat/multilingual-e5-large
```

The model is loaded automatically through `sentence-transformers` and does not need to be manually placed in the repository.

Model:

https://huggingface.co/intfloat/multilingual-e5-large

### Prometheus 2

The explanation evaluation uses:

```text
prometheus-7b-v2.0.Q4_K_M.gguf
```

Download:

https://huggingface.co/prometheus-eval/prometheus-7b-v2.0-GGUF/blob/main/prometheus-7b-v2.0.Q4_K_M.gguf

Place the downloaded file in:

```text
src/explanation_generation/
```

so that the final path is:

```text
src/explanation_generation/prometheus-7b-v2.0.Q4_K_M.gguf
```

## Dependencies

The main Python dependencies are:

```text
pandas
numpy
scikit-learn
sentence-transformers
llama-cpp-python
pynvml
openpyxl
```

## Execution

The official experimental sample is already included in the repository. To validate it:

```bash
python create_sample.py
```

To run the explanation-generation pipeline:

```bash
python run_pipeline.py
```

This produces or updates:

```text
pipeline_results_1000.xlsx
```

To evaluate the generated explanations:

```bash
python evaluate_explanations.py
```

This produces or updates:

```text
evaluation_results_1000.xlsx
```