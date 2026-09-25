# Evidence Coverage

This module evaluates different retrieval strategies for identifying evidence relevant to individual fake news items. For each fake news item, the retrieval stage returns the Top-10 candidate evidences, which are then provided to Mistral-Nemo for the final evidence-selection stage.

The experiments are conducted on a benchmark of 101 fake news items using three progressively extended evidence sets.

## Experimental Setup

Three evidence sets are evaluated:

- **54 Analytical Validations** — evidence identified as `VALIDAZIONE ANALITICA` in the single-label categorization experiment.
- **90 Analytical Validations** — extended analytical-validation set obtained by also considering the double-label categorization, including evidence where `VALIDAZIONE ANALITICA` appears as one of the assigned labels.
- **319 CTI Evidences** — complete evidence corpus containing all CTI categories.

The same 101 fake news items are used across the experiments. However, each evidence set has its own ground truth, since the evidence available for covering each fake news item changes with the corpus.

## Pipeline

For each evidence set, the coverage experiment follows the same pipeline:

1. A benchmark of **101 fake news items** is used as the input set.
2. One of the four retrieval strategies (**Keyword, TF-IDF, E5, or Cross-Encoder**) ranks the available evidences and retrieves the **Top-10 candidates** for each fake news item.
3. The fake news item and its Top-10 retrieved candidates are provided to **Mistral-Nemo-Instruct-2407-Q4_K_S**.
4. Mistral-Nemo performs the evidence-selection stage, returning the IDs of the relevant evidences or `NONE` when no candidate satisfies the selection criteria.
5. The resulting output is evaluated against the ground truth associated with the selected evidence set through the common `evaluate.py` script.
6. `evaluate.py` computes metrics for both the retrieval stage and the subsequent evidence-selection stage.

The pipeline is repeated for each combination of the **3 evidence sets × 4 retrieval strategies**, resulting in 12 experimental configurations.

## Retrieval Strategies

Four retrieval approaches are evaluated for each evidence set:

- **Keyword Matching**
- **TF-IDF**
- **E5** (`intfloat/multilingual-e5-large`)
- **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`)

Each strategy ranks the available evidences and retrieves the **Top-10 candidates** for each fake news item.

The retrieved candidates are then provided to **Mistral-Nemo-Instruct-2407-Q4_K_S**, which selects only the evidence that provides a concrete connection to the specific fake news claim, source, event, actor, or documented disinformation operation. If no candidate satisfies the selection criteria, the model returns `NONE`.

## Structure

```text
src/coverage/
├── README.md
├── Coverage_All_Results.xlsx
├── evaluate.py
├── validations_54/
│   ├── cross_encoder.py
│   ├── keyword.py
│   ├── tfidf.py
│   └── e5.py
├── validations_90/
│   ├── cross_encoder.py
│   ├── keyword.py
│   ├── tfidf.py
│   └── e5.py
└── evidences_319/
    ├── cross_encoder.py
    ├── keyword.py
    ├── tfidf.py
    └── e5.py
```

The corresponding datasets are stored in:

```text
data/coverage/
├── Fake101_Benchmark.xlsx
├── Analytical_Validations_54.xlsx
├── Analytical_Validations_90.xlsx
├── CTI_Evidences_319.xlsx
├── Fake101_Ground_Truth_54.xlsx
├── Fake101_Ground_Truth_90.xlsx
└── Fake101_Ground_Truth_319.xlsx
```

## Evaluation

`evaluate.py` provides a common evaluation procedure for all 12 experimental configurations.

The evaluation includes:

- Retrieval Hit Rate@10
- Mean Precision@10
- Mean Recall@10
- Mean F1@10
- LLM evidence-selection Precision, Recall and F1
- Micro Precision, Recall and F1
- `NONE` Accuracy
- Exact Match Accuracy
- Overall Mean F1 per fake news item

Retrieval metrics are computed only for fake news items with at least one relevant evidence in the corresponding ground truth, since the retrieval stage always returns a Top-10 candidate list.

To evaluate a specific configuration, select the corresponding evidence set, ground-truth file, and retrieval output in the configuration section at the bottom of `evaluate.py`.

For example:

```python
EXPERIMENT_DIR = SCRIPT_DIR / "evidences_319"

MANUAL_FILE = DATA_DIR / "Fake101_Ground_Truth_319.xlsx"
MODEL_FILE = EXPERIMENT_DIR / "CrossEncoder_Fake101.xlsx"
OUTPUT_FILE = EXPERIMENT_DIR / "Report_CrossEncoder_Fake101.xlsx"
```

The ground-truth file must always correspond to the selected evidence set:

```text
validations_54 -> Fake101_Ground_Truth_54.xlsx
validations_90 -> Fake101_Ground_Truth_90.xlsx
evidences_319  -> Fake101_Ground_Truth_319.xlsx
```

## Results

`Coverage_All_Results.xlsx` contains the detailed results and summary metrics for the complete **3 evidence sets × 4 retrieval methods** experimental matrix.

The `Comparison` sheet provides the aggregated comparison across the 12 configurations.

## Models

The experiments use:

- **Mistral-Nemo-Instruct-2407-Q4_K_S** for evidence selection — [Hugging Face](https://huggingface.co/mradermacher/Mistral-Nemo-Instruct-2407-GGUF)
- **multilingual-e5-large** for dense retrieval — automatically downloaded through `sentence-transformers`
- **ms-marco-MiniLM-L-6-v2** for Cross-Encoder retrieval — automatically downloaded through `sentence-transformers`