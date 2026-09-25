from pathlib import Path
import re

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment


# ============================================================
# ID PARSING
# ============================================================
def parse_ids(text):
    """
    Parse evidence IDs from a cell and return them as a normalized set.

    Supported examples:
        "V01, V05"       -> {"V01", "V05"}
        "CT201, AN199"   -> {"CT201", "AN199"}
        "NONE"           -> set()
    """
    if pd.isna(text):
        return set()

    text_str = str(text).strip().upper()

    if text_str in {"NONE", "", "NAN", "NAT"}:
        return set()

    # Supports both validation IDs used in the 54/90 experiments
    # and category-coded evidence IDs used in the 319 experiment.
    found = re.findall(
        r"\b(?:V\d{2,3}|[A-Z]{1,2}\d{3})\b",
        text_str,
    )

    return set(found)


# ============================================================
# METRIC UTILITIES
# ============================================================
def safe_f1(precision, recall):
    """Return the harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# ============================================================
# EVALUATION
# ============================================================
def evaluate_predictions(manual_file, model_file, output_report_file):
    print(f"Loading manual ground-truth file: {manual_file}")
    df_manual = pd.read_excel(manual_file)

    print(f"Loading model output file: {model_file}")
    df_model = pd.read_excel(model_file)

    # Normalize column names
    df_manual.columns = df_manual.columns.str.strip()
    df_model.columns = df_model.columns.str.strip()

    if "ID_FAKE" not in df_manual.columns or "ID_FAKE" not in df_model.columns:
        raise ValueError(
            "The 'ID_FAKE' column must be present in both input files."
        )

    # Merge ground truth and model output by fake-news ID
    print("Merging datasets on 'ID_FAKE'...")

    df_merged = pd.merge(
        df_manual,
        df_model,
        on="ID_FAKE",
        how="inner",
    )

    if len(df_merged) != len(df_manual):
        print(
            "WARNING: the merged dataset does not contain the same number "
            "of rows as the manual ground-truth file."
        )

    # Automatically identify the manual evidence column
    possible_val_cols = [
        col
        for col in df_manual.columns
        if "validazione" in col.lower()
        or "id_val" in col.lower()
        or "validation" in col.lower()
    ]

    col_manual_val = (
        possible_val_cols[0]
        if possible_val_cols
        else "VALIDAZIONE MANUALE"
    )

    if col_manual_val not in df_merged.columns:
        raise ValueError(
            f"Ground-truth validation column not found: '{col_manual_val}'."
        )

    required_model_columns = {
        "RISPOSTA_RETRIEVAL",
        "ID_VALIDAZIONE",
    }

    missing_columns = (
        required_model_columns - set(df_merged.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required model-output columns: "
            + ", ".join(sorted(missing_columns))
        )

    detailed_results = []

    # Retrieval accumulators:
    # only fake news with non-empty ground truth
    retrieval_hits = 0
    retrieval_precisions = []
    retrieval_recalls = []
    retrieval_f1s = []

    # LLM evidence-selection accumulators:
    # only fake news with non-empty ground truth
    evidence_precisions = []
    evidence_recalls = []
    evidence_f1s = []

    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    # Global pipeline accumulators
    none_total = 0
    none_correct = 0
    exact_matches = 0
    overall_instance_f1s = []

    # ========================================================
    # PER-FAKE EVALUATION
    # ========================================================
    for _, row in df_merged.iterrows():
        id_fake = row["ID_FAKE"]

        # Descriptive fields from the manual dataset
        campaign = row.get(
            "CAMPAIGN",
            row.get("Campaign", ""),
        )

        title = row.get(
            "TITLE",
            row.get(
                "Title",
                row.get("TITOLO", ""),
            ),
        )

        fake_text = row.get(
            "TEXT",
            row.get("Text", ""),
        )

        # Ground truth
        true_str = str(
            row.get(
                col_manual_val,
                "NONE",
            )
        )

        true_set = parse_ids(true_str)
        is_covered = bool(true_set)

        # Retrieval output
        retrieval_str = str(
            row.get(
                "RISPOSTA_RETRIEVAL",
                "",
            )
        )

        retrieval_set = parse_ids(retrieval_str)

        # LLM evidence-selection output
        pred_str = str(
            row.get(
                "ID_VALIDAZIONE",
                "NONE",
            )
        )

        pred_set = parse_ids(pred_str)

        # ----------------------------------------------------
        # RETRIEVAL METRICS
        # ----------------------------------------------------
        # Evaluated only on covered fake news (GT != NONE),
        # because retrieval always returns a Top-K list and
        # cannot predict NONE.
        if is_covered:
            tp_ret = len(true_set & retrieval_set)
            fp_ret = len(retrieval_set - true_set)
            fn_ret = len(true_set - retrieval_set)

            ret_precision = (
                tp_ret / (tp_ret + fp_ret)
                if (tp_ret + fp_ret) > 0
                else 0.0
            )

            ret_recall = (
                tp_ret / (tp_ret + fn_ret)
            )

            ret_f1 = safe_f1(
                ret_precision,
                ret_recall,
            )

            ret_hit = tp_ret > 0

            retrieval_precisions.append(
                ret_precision
            )

            retrieval_recalls.append(
                ret_recall
            )

            retrieval_f1s.append(
                ret_f1
            )

            retrieval_hits += int(
                ret_hit
            )

        else:
            # NONE cases are intentionally excluded
            # from retrieval metrics.
            ret_precision = np.nan
            ret_recall = np.nan
            ret_f1 = np.nan
            ret_hit = None

        # ----------------------------------------------------
        # LLM EVIDENCE-SELECTION METRICS
        # ----------------------------------------------------
        # Mean-per-fake and micro metrics are evaluated only
        # on covered fake news (GT != NONE).
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        if is_covered:
            # If the model predicts NONE for a covered fake,
            # precision, recall and F1 are all defined as 0.
            evidence_precision = (
                tp / (tp + fp)
                if (tp + fp) > 0
                else 0.0
            )

            evidence_recall = (
                tp / (tp + fn)
            )

            evidence_f1 = safe_f1(
                evidence_precision,
                evidence_recall,
            )

            evidence_precisions.append(
                evidence_precision
            )

            evidence_recalls.append(
                evidence_recall
            )

            evidence_f1s.append(
                evidence_f1
            )

            micro_tp += tp
            micro_fp += fp
            micro_fn += fn

            # For covered cases, the overall instance score
            # is the evidence-selection F1 for that fake news.
            overall_instance_f1 = evidence_f1

        else:
            # Evidence P/R/F1 are not defined for GT=NONE
            # in the evidence-identification evaluation.
            evidence_precision = np.nan
            evidence_recall = np.nan
            evidence_f1 = np.nan

            none_total += 1

            correct_none = not pred_set
            none_correct += int(correct_none)

            # Task-specific convention for the global metric:
            # correct NONE -> 1
            # incorrect NONE -> 0
            overall_instance_f1 = (
                1.0 if correct_none else 0.0
            )

        overall_instance_f1s.append(
            overall_instance_f1
        )

        # Exact Match Accuracy is evaluated on every fake news.
        is_exact = true_set == pred_set
        exact_matches += int(is_exact)

        detailed_results.append(
            {
                "ID_FAKE": id_fake,
                "CAMPAIGN": campaign,
                "TITLE": title,
                "TEXT": fake_text,
                "RETRIEVAL_OUTPUT": retrieval_str,
                "GROUND_TRUTH": true_str,
                "MODEL_OUTPUT": pred_str,
                "GT_COVERED": (
                    "YES" if is_covered else "NO"
                ),
                "RETRIEVAL_HIT_AT_10": (
                    "YES"
                    if ret_hit is True
                    else "NO"
                    if ret_hit is False
                    else "N/A"
                ),
                "RETRIEVAL_PRECISION_AT_10": ret_precision,
                "RETRIEVAL_RECALL_AT_10": ret_recall,
                "RETRIEVAL_F1_AT_10": ret_f1,
                "LLM_EVIDENCE_PRECISION": evidence_precision,
                "LLM_EVIDENCE_RECALL": evidence_recall,
                "LLM_EVIDENCE_F1": evidence_f1,
                "OVERALL_INSTANCE_F1": overall_instance_f1,
                "EXACT_MATCH": (
                    1.0 if is_exact else 0.0
                ),
            }
        )

    df_report = pd.DataFrame(
        detailed_results
    )

    n_total = len(df_report)
    n_covered = len(evidence_f1s)

    # ========================================================
    # GLOBAL RETRIEVAL METRICS
    # ========================================================
    retrieval_hit_rate_at_10 = (
        retrieval_hits / n_covered
        if n_covered > 0
        else 0.0
    )

    mean_ret_precision_at_10 = (
        float(np.mean(retrieval_precisions))
        if retrieval_precisions
        else 0.0
    )

    mean_ret_recall_at_10 = (
        float(np.mean(retrieval_recalls))
        if retrieval_recalls
        else 0.0
    )

    mean_ret_f1_at_10 = (
        float(np.mean(retrieval_f1s))
        if retrieval_f1s
        else 0.0
    )

    # ========================================================
    # GLOBAL LLM EVIDENCE-SELECTION METRICS
    # ========================================================
    mean_evidence_precision = (
        float(np.mean(evidence_precisions))
        if evidence_precisions
        else 0.0
    )

    mean_evidence_recall = (
        float(np.mean(evidence_recalls))
        if evidence_recalls
        else 0.0
    )

    mean_evidence_f1 = (
        float(np.mean(evidence_f1s))
        if evidence_f1s
        else 0.0
    )

    micro_precision = (
        micro_tp / (micro_tp + micro_fp)
        if (micro_tp + micro_fp) > 0
        else 0.0
    )

    micro_recall = (
        micro_tp / (micro_tp + micro_fn)
        if (micro_tp + micro_fn) > 0
        else 0.0
    )

    micro_f1 = safe_f1(
        micro_precision,
        micro_recall,
    )

    # ========================================================
    # NONE AND END-TO-END METRICS
    # ========================================================
    none_accuracy = (
        none_correct / none_total
        if none_total > 0
        else 0.0
    )

    exact_match_accuracy = (
        exact_matches / n_total
        if n_total > 0
        else 0.0
    )

    overall_mean_f1_per_fake = (
        float(np.mean(overall_instance_f1s))
        if overall_instance_f1s
        else 0.0
    )

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================
    print("\n" + "=" * 72)
    print("OVERALL MODEL & RETRIEVAL EVALUATION")
    print("=" * 72)

    print(
        f"Total fake news                     : "
        f"{n_total}"
    )

    print(
        f"Covered fake news (GT != NONE)      : "
        f"{n_covered}"
    )

    print(
        f"NONE fake news (GT = NONE)          : "
        f"{none_total}"
    )

    print("-" * 72)
    print(
        "RETRIEVAL METRICS - COVERED FAKE NEWS ONLY"
    )

    print(
        f"Hit Rate@10                         : "
        f"{retrieval_hit_rate_at_10:.4f} "
        f"({retrieval_hits}/{n_covered})"
    )

    print(
        f"Mean Recall@10                      : "
        f"{mean_ret_recall_at_10:.4f}"
    )

    print(
        f"Mean Precision@10                   : "
        f"{mean_ret_precision_at_10:.4f}"
    )

    print(
        f"Mean F1@10                          : "
        f"{mean_ret_f1_at_10:.4f}"
    )

    print("-" * 72)
    print(
        "LLM EVIDENCE SELECTION - COVERED FAKE NEWS ONLY"
    )

    print(
        f"Mean Precision per fake             : "
        f"{mean_evidence_precision:.4f}"
    )

    print(
        f"Mean Recall per fake                : "
        f"{mean_evidence_recall:.4f}"
    )

    print(
        f"Mean F1 per fake                    : "
        f"{mean_evidence_f1:.4f}"
    )

    print(
        f"Micro Precision                     : "
        f"{micro_precision:.4f}"
    )

    print(
        f"Micro Recall                        : "
        f"{micro_recall:.4f}"
    )

    print(
        f"Micro F1                            : "
        f"{micro_f1:.4f}"
    )

    print(
        f"Micro counts (TP / FP / FN)         : "
        f"{micro_tp} / {micro_fp} / {micro_fn}"
    )

    print("-" * 72)
    print(
        "NONE AND END-TO-END PIPELINE METRICS"
    )

    print(
        f"NONE Accuracy                       : "
        f"{none_accuracy:.4f} "
        f"({none_correct}/{none_total})"
    )

    print(
        f"Exact Match Accuracy                : "
        f"{exact_match_accuracy:.4f} "
        f"({exact_matches}/{n_total})"
    )

    print(
        f"Overall Mean F1 per fake            : "
        f"{overall_mean_f1_per_fake:.4f}"
    )

    print("=" * 72)

    # ========================================================
    # EXCEL REPORT
    # ========================================================
    with pd.ExcelWriter(
        output_report_file,
        engine="openpyxl",
    ) as writer:

        df_report.to_excel(
            writer,
            sheet_name="Record_Details",
            index=False,
        )

        summary_rows = [
            [
                "DATASET",
                "Total fake news",
                n_total,
                "All",
            ],
            [
                "DATASET",
                "Covered fake news",
                n_covered,
                "GT != NONE",
            ],
            [
                "DATASET",
                "NONE fake news",
                none_total,
                "GT = NONE",
            ],
            [
                "RETRIEVAL",
                "Hit Rate@10",
                retrieval_hit_rate_at_10,
                "GT != NONE",
            ],
            [
                "RETRIEVAL",
                "Mean Recall@10",
                mean_ret_recall_at_10,
                "GT != NONE",
            ],
            [
                "RETRIEVAL",
                "Mean Precision@10",
                mean_ret_precision_at_10,
                "GT != NONE",
            ],
            [
                "RETRIEVAL",
                "Mean F1@10",
                mean_ret_f1_at_10,
                "GT != NONE",
            ],
            [
                "LLM",
                "Mean Precision per fake",
                mean_evidence_precision,
                "GT != NONE",
            ],
            [
                "LLM",
                "Mean Recall per fake",
                mean_evidence_recall,
                "GT != NONE",
            ],
            [
                "LLM",
                "Mean F1 per fake",
                mean_evidence_f1,
                "GT != NONE",
            ],
            [
                "LLM",
                "Micro Precision",
                micro_precision,
                "GT != NONE",
            ],
            [
                "LLM",
                "Micro Recall",
                micro_recall,
                "GT != NONE",
            ],
            [
                "LLM",
                "Micro F1",
                micro_f1,
                "GT != NONE",
            ],
            [
                "LLM",
                "Micro TP",
                micro_tp,
                "GT != NONE",
            ],
            [
                "LLM",
                "Micro FP",
                micro_fp,
                "GT != NONE",
            ],
            [
                "LLM",
                "Micro FN",
                micro_fn,
                "GT != NONE",
            ],
            [
                "LLM",
                "NONE Accuracy",
                none_accuracy,
                "GT = NONE",
            ],
            [
                "PIPELINE",
                "Exact Match Accuracy",
                exact_match_accuracy,
                "All",
            ],
            [
                "PIPELINE",
                "Overall Mean F1 per fake",
                overall_mean_f1_per_fake,
                "All",
            ],
        ]

        df_summary = pd.DataFrame(
            summary_rows,
            columns=[
                "Component",
                "Metric",
                "Value",
                "Evaluation cases",
            ],
        )

        df_summary.to_excel(
            writer,
            sheet_name="Global_Summary",
            index=False,
        )

        # Basic formatting for readability
        for sheet_name in [
            "Record_Details",
            "Global_Summary",
        ]:
            worksheet = writer.sheets[
                sheet_name
            ]

            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = (
                worksheet.dimensions
            )

            for cell in worksheet[1]:
                cell.alignment = Alignment(
                    horizontal="center",
                    vertical="center",
                )

        summary_ws = writer.sheets[
            "Global_Summary"
        ]

        summary_ws.column_dimensions["A"].width = 16
        summary_ws.column_dimensions["B"].width = 32
        summary_ws.column_dimensions["C"].width = 16
        summary_ws.column_dimensions["D"].width = 18

    print(
        f"\nComplete report saved to: "
        f"{output_report_file}"
    )

    return df_report


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================
if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parents[1]
    DATA_DIR = PROJECT_ROOT / "data" / "coverage"

    # Select the evidence set and retrieval method to evaluate.
    #
    # Evidence sets:
    #   validations_54 -> Fake101_Ground_Truth_54.xlsx
    #   validations_90 -> Fake101_Ground_Truth_90.xlsx
    #   evidences_319  -> Fake101_Ground_Truth_319.xlsx
    #
    # Retrieval outputs:
    #   CrossEncoder_Fake101.xlsx
    #   Keyword_Fake101.xlsx
    #   TFIDF_Fake101.xlsx
    #   E5_Fake101.xlsx

    EXPERIMENT_DIR = SCRIPT_DIR / "evidences_319"

    MANUAL_FILE = (
        DATA_DIR
        / "Fake101_Ground_Truth_319.xlsx"
    )

    MODEL_FILE = (
        EXPERIMENT_DIR
        / "CrossEncoder_Fake101.xlsx"
    )

    OUTPUT_FILE = (
        EXPERIMENT_DIR
        / "Report_CrossEncoder_Fake101.xlsx"
    )

    evaluate_predictions(
        MANUAL_FILE,
        MODEL_FILE,
        OUTPUT_FILE,
    )