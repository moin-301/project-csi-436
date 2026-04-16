# =============================================================================
# evaluator.py
#
# PURPOSE : Evaluate a trained model and save every output needed
#           for later plotting and comparison.
#           This module does ONE job only -- measuring and saving results.
#           It does NOT train, search, or plot anything.
#
# METRICS COMPUTED:
#   Core        : Accuracy, Precision, Recall, F1 (weighted), MCC
#   Advanced    : ROC-AUC, PR-AUC  (via decision_function scores)
#   Efficiency  : Inference time, Number of support vectors
#   Detailed    : Per-class precision/recall/f1, confusion matrix
#
# FILES IT SAVES (inside the run folder):
#   metrics.json          -- all numeric scores
#   y_test.npy            -- true labels
#   y_pred.npy            -- predicted labels  (for confusion matrix)
#   y_scores.npy          -- decision function scores  (for ROC and PR curves)
#   classification_report.txt -- full per-class text report
# =============================================================================

import time
import json
import logging
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("evaluator")


# =============================================================================
# EVALUATE AND SAVE
# =============================================================================

def evaluate_and_save(model, X_test, y_test, run_folder: Path, metadata: dict):
    """
    Run the model on test data, compute all metrics, and save everything
    needed for later plotting and comparison tables.

    Parameters
    ----------
    model      : trained sklearn SVC
    X_test     : sparse TF-IDF matrix -- test features
    y_test     : numpy array          -- true labels (0 or 1)
    run_folder : Path                 -- the timestamped folder for this run
    metadata   : dict                 -- from trainer.py, will be extended here

    Returns
    -------
    metrics : dict -- all computed scores

    Example
    -------
    metrics = evaluate_and_save(model, X_test, y_test, run_folder, metadata)
    """

    kernel     = metadata.get("kernel", "unknown")
    model_name = run_folder.name

    logger.info(f"Evaluating {model_name}...")
    print(f"\n  {'='*55}")
    print(f"  Evaluating : {model_name}")
    print(f"  {'='*55}")

    # ------------------------------------------------------------------
    # Step 1 -- Predictions
    # ------------------------------------------------------------------

    infer_start = time.time()
    y_pred      = model.predict(X_test)
    infer_time  = time.time() - infer_start

    # Decision function scores -- used for ROC and PR curves
    # Works without probability=True, so training stays fast
    y_scores = model.decision_function(X_test)

    # ------------------------------------------------------------------
    # Step 2 -- Core Metrics
    # ------------------------------------------------------------------

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    mcc       = matthews_corrcoef(y_test, y_pred)

    # Per-class metrics (for the report)
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0).tolist()
    recall_per_class    = recall_score(y_test, y_pred, average=None, zero_division=0).tolist()
    f1_per_class        = f1_score(y_test, y_pred, average=None, zero_division=0).tolist()

    # ------------------------------------------------------------------
    # Step 3 -- Advanced Metrics (ROC-AUC and PR-AUC)
    # ------------------------------------------------------------------

    roc_auc = roc_auc_score(y_test, y_scores)
    pr_auc  = average_precision_score(y_test, y_scores)

    # ------------------------------------------------------------------
    # Step 4 -- Efficiency
    # ------------------------------------------------------------------

    num_support_vectors = int(model.n_support_.sum()) if hasattr(model, "n_support_") else None
    training_time       = metadata.get("training_time_seconds", None)

    # ------------------------------------------------------------------
    # Step 5 -- Confusion Matrix
    # ------------------------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    # ------------------------------------------------------------------
    # Step 6 -- Full Classification Report (text)
    # ------------------------------------------------------------------

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=["Negative", "Positive"],
        zero_division=0,
    )

    # ------------------------------------------------------------------
    # Step 7 -- Pack everything into one dictionary
    # ------------------------------------------------------------------

    metrics = {
        # Identification
        "model_name"            : model_name,
        "kernel"                : kernel,
        "timestamp"             : metadata.get("timestamp", ""),
        "hyperparameters"       : metadata.get("hyperparameters", {}),

        # Core metrics
        "accuracy"              : round(float(accuracy),  4),
        "precision_weighted"    : round(float(precision), 4),
        "recall_weighted"       : round(float(recall),    4),
        "f1_weighted"           : round(float(f1),        4),
        "mcc"                   : round(float(mcc),       4),

        # Per-class metrics
        "precision_per_class"   : [round(v, 4) for v in precision_per_class],
        "recall_per_class"      : [round(v, 4) for v in recall_per_class],
        "f1_per_class"          : [round(v, 4) for v in f1_per_class],
        "class_names"           : ["Negative", "Positive"],

        # Advanced metrics
        "roc_auc"               : round(float(roc_auc), 4),
        "pr_auc"                : round(float(pr_auc),  4),

        # Efficiency
        "training_time_seconds" : round(float(training_time), 2) if training_time else None,
        "inference_time_seconds": round(float(infer_time), 4),
        "num_support_vectors"   : num_support_vectors,

        # Data info
        "test_samples"          : int(len(y_test)),
        "num_features"          : int(X_test.shape[1]),

        # Confusion matrix
        "confusion_matrix"      : cm.tolist(),
    }

    # ------------------------------------------------------------------
    # Step 8 -- Print to console
    # ------------------------------------------------------------------

    _print_metrics(metrics, report_text)

    # ------------------------------------------------------------------
    # Step 9 -- Save everything to the run folder
    # ------------------------------------------------------------------

    _save_all(
        run_folder  = run_folder,
        metrics     = metrics,
        report_text = report_text,
        y_test      = y_test,
        y_pred      = y_pred,
        y_scores    = y_scores,
    )

    logger.info(f"Evaluation complete | {model_name}")

    return metrics


# =============================================================================
# PRINT HELPER
# =============================================================================

def _print_metrics(metrics: dict, report_text: str):
    """Print a clean summary of all metrics to the console."""

    divider = "=" * 55

    print(f"\n  {divider}")
    print(f"  Results: {metrics['model_name']}")
    print(f"  {divider}")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Precision (w)     : {metrics['precision_weighted']:.4f}")
    print(f"  Recall    (w)     : {metrics['recall_weighted']:.4f}")
    print(f"  F1        (w)     : {metrics['f1_weighted']:.4f}")
    print(f"  MCC               : {metrics['mcc']:.4f}")
    print(f"  ROC-AUC           : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC            : {metrics['pr_auc']:.4f}")
    print(f"  Training time     : {metrics['training_time_seconds']}s")
    print(f"  Inference time    : {metrics['inference_time_seconds']}s")
    print(f"  Support vectors   : {metrics['num_support_vectors']}")
    print(f"  {divider}")

    cm = np.array(metrics["confusion_matrix"])
    print(f"  Confusion Matrix:")
    print(f"    {'':14}  Pred NEG   Pred POS")
    print(f"    {'True NEG':14}  {cm[0][0]:<10} {cm[0][1]}")
    print(f"    {'True POS':14}  {cm[1][0]:<10} {cm[1][1]}")
    print(f"  {divider}")

    print(f"\n  Classification Report:")
    for line in report_text.splitlines():
        print(f"    {line}")

    print(f"  {divider}\n")


# =============================================================================
# SAVE HELPER
# =============================================================================

def _save_all(run_folder, metrics, report_text, y_test, y_pred, y_scores):
    """
    Save all outputs into the run folder.

    Files saved:
      metrics.json                -- all numeric scores (JSON)
      classification_report.txt   -- full per-class text report
      y_test.npy                  -- true labels
      y_pred.npy                  -- predicted labels
      y_scores.npy                -- decision function scores
    """

    # metrics.json
    metrics_path = run_folder / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved           : {metrics_path}")
    print(f"  Metrics saved           : metrics.json")

    # classification_report.txt
    report_path = run_folder / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    logger.info(f"Report saved            : {report_path}")
    print(f"  Report saved            : classification_report.txt")

    # y_test.npy
    y_test_path = run_folder / "y_test.npy"
    np.save(str(y_test_path), y_test)
    logger.info(f"y_test saved            : {y_test_path}")
    print(f"  y_test saved            : y_test.npy")

    # y_pred.npy
    y_pred_path = run_folder / "y_pred.npy"
    np.save(str(y_pred_path), y_pred)
    logger.info(f"y_pred saved            : {y_pred_path}")
    print(f"  y_pred saved            : y_pred.npy")

    # y_scores.npy  (decision function -- for ROC and PR curves)
    y_scores_path = run_folder / "y_scores.npy"
    np.save(str(y_scores_path), y_scores)
    logger.info(f"y_scores saved          : {y_scores_path}")
    print(f"  y_scores saved          : y_scores.npy")
