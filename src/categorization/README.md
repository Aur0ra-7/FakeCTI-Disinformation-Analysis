# CTI EVIDENCE CATEGORIZATION

This module contains the experiments performed to automatically categorize
CTI evidence collected from the FakeCTI disinformation campaigns using local
Large Language Models (LLMs).

Two models were evaluated using different prompt configurations and temperature
values. The generated predictions were compared with manually assigned reference
categories.

## MODELS

### Llama 3 8B

- Model: Llama-3-8B-Instruct-v0.10
- Quantization: `Q5_K_M`
- Model file: `Llama-3-8B-Instruct-v0.10.Q5_K_M.gguf`
- Hugging Face: [Llama-3-8B-Instruct-v0.10-GGUF](https://huggingface.co/MaziyarPanahi/Llama-3-8B-Instruct-v0.10-GGUF)

### Mistral-Nemo

- Model: Mistral-Nemo-Instruct-2407
- Quantization: `Q4_K_S`
- Model file: `Mistral-Nemo-Instruct-2407-Q4_K_S.gguf`
- Hugging Face: [Mistral-Nemo-Instruct-2407-GGUF](https://huggingface.co/mradermacher/Mistral-Nemo-Instruct-2407-GGUF)

The GGUF model files are not included in this repository and must be downloaded
separately from the Hugging Face repositories linked above.


## EXPERIMENTAL WORKFLOW

For both models, the experiment consists of two steps:

1. **Categorization** — generate `primary` and `secondary` category predictions for each CTI evidence.
2. **Evaluation** — compare the predictions with the manually assigned reference categories.

### Llama 3 8B

```bash
python Categorization.py
python Evaluate_Categorization.py
```

### Mistral-Nemo

```bash
python categorization.py
python evaluate_categorization.py
```

In both cases, the output produced by the categorization script is used as
input to the corresponding evaluation script.


## PROMPT AND TEMPERATURE CONFIGURATIONS

Different prompts were tested for both models using the following temperatures:

`0.0` · `0.3` · `0.6` · `0.9`

The tested prompts are available in `provaprompt.txt`.

Prompt and temperature configurations are selected manually before each run.

> **Important:** Input and output filenames must be updated between experimental
> configurations. Reusing the same output filename will overwrite the results
> of the previous run. Each categorization output must also be provided as input
> to the corresponding evaluation run.


## DOUBLE CATEGORIZATION

An additional evaluation was performed using an extended multi-label ground truth.
The original manual annotation assigned one reference category to each CTI evidence.
The ground truth was subsequently extended by adding a second reference category
when an evidence contained two distinct informational aspects.

Instead of repeating the complete model, prompt, and temperature exploration,
this experiment uses the best-performing Mistral-Nemo configuration identified
during the previous experiments:

- Model: `Mistral-Nemo-Instruct-2407-Q4_K_S`
- Prompt: `Prompt 4`
- Temperature: `0.3`

`MultiLabel.xlsx` contains the predictions generated using this configuration
and is provided so that the evaluation can be reproduced without rerunning the
LLM inference.

Run:

```bash
python double_evaluate.py
```

The resulting evaluation is stored in:

```text
MultiLabel_Evaluation.xlsx
```
## REPOSITORY STRUCTURE

    categorization/
    |
    |-- README.md
    |-- provaprompt.txt
    |-- SingleLabel_All_Results.xlsx
    |
    |-- Llama_3_8B/
    |   |-- Categorization.py
    |   `-- Evaluate_Categorization.py
    |
    `-- Mistral_Nemo/
        |-- categorization.py
        |-- evaluate_categorization.py
        |
        `-- Double_categorization/
            |-- double_evaluate.py
            |-- MultiLabel.xlsx
            `-- MultiLabel_Evaluation.xlsx

## OUTPUT FILES

- `SingleLabel_All_Results.xlsx` — aggregated single-label experimental results.
- `MultiLabel.xlsx` — predictions generated with the selected Mistral-Nemo configuration.
- `MultiLabel_Evaluation.xlsx` — double-categorization evaluation results.


## REPRODUCIBILITY

Before running the experiments, configure the local model path and the required
input/output paths in the corresponding scripts.

Each single-label prompt/temperature configuration must be executed separately
and its output preserved before running the corresponding evaluation.