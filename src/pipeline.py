# =============================================================================
# pipeline.py
#
# PURPOSE : The ONLY file you need to run.
#           Reads config, loads data, trains, evaluates, and saves everything.
#
# HOW TO RUN:
#   python pipeline.py
#
# TO TRAIN ONE KERNEL:
#   Open config.py and set:
#     TRAIN_KERNEL = "rbf"
#
# TO TRAIN ALL KERNELS:
#   Open config.py and set:
#     TRAIN_KERNEL = "all"
#
# TO ADD A NEW KERNEL:
#   Open config.py and add the kernel name to ALL_KERNELS
#   and its default parameters to DEFAULT_PARAMS.
#   Nothing else needs to change.
#
# OUTPUT STRUCTURE:
#   results/
#     svm_rbf_2025-04-15_1430/
#       model.pkl
#       hyperparameters.json
#       metadata.json
#       metrics.json
#       classification_report.txt
#       y_test.npy
#       y_pred.npy
#       y_scores.npy
#   logs/
#     pipeline.log
# =============================================================================

import sys
import time
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure imports work when running from any directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from config import (
    TRAIN_KERNEL,
    ALL_KERNELS,
    DEFAULT_PARAMS,
    LOG_FORMAT,
    LOG_DATE_FMT,
)
from paths import (
    LOGS_DIR,
    LOG_FILE,
    RESULTS_DIR,
)
from data_loader import load_all
from trainer     import run_training
from evaluator   import evaluate_and_save


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """
    Set up logging for the whole pipeline.
    Logs go to both the terminal and logs/pipeline.log.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level    = logging.INFO,
        format   = LOG_FORMAT,
        datefmt  = LOG_DATE_FMT,
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE),
        ],
    )


logger = logging.getLogger("pipeline")


# =============================================================================
# TRAIN AND EVALUATE ONE KERNEL
# =============================================================================

def run_one_kernel(kernel: str, X_train, y_train, X_test, y_test):
    """
    Full pipeline for a single kernel:
      1. Train with default hyperparameters from config
      2. Evaluate on test set
      3. Save everything

    Parameters
    ----------
    kernel           : str
    X_train, y_train : training data
    X_test,  y_test  : test data

    Returns
    -------
    metrics : dict -- all evaluation scores for this run
    """

    print(f"\n  {'#'*55}")
    print(f"  KERNEL : {kernel.upper()}")
    print(f"  Params : {DEFAULT_PARAMS.get(kernel, {})}")
    print(f"  {'#'*55}")

    logger.info(f"Starting run | kernel={kernel} | params={DEFAULT_PARAMS.get(kernel, {})}")

    # Train and save model
    model, metadata, run_folder = run_training(
        kernel  = kernel,
        X_train = X_train,
        y_train = y_train,
    )

    # Evaluate and save all outputs
    metrics = evaluate_and_save(
        model      = model,
        X_test     = X_test,
        y_test     = y_test,
        run_folder = run_folder,
        metadata   = metadata,
    )

    logger.info(f"Run complete | kernel={kernel} | folder={run_folder.name}")

    return metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Main pipeline entry point.

    Reads TRAIN_KERNEL from config.py:
      "all"  -- trains every kernel in ALL_KERNELS
      "rbf"  -- trains only rbf  (or any other single kernel name)
    """

    setup_logging()

    total_start = time.time()

    # ---- Banner ----
    print("\n")
    print("  " + "="*55)
    print("  SVM SENTIMENT ANALYSIS -- TRAINING PIPELINE")
    print("  " + "="*55)

    if TRAIN_KERNEL == "all":
        print(f"  Mode    : Train ALL kernels")
        print(f"  Kernels : {ALL_KERNELS}")
    else:
        print(f"  Mode    : Train ONE kernel")
        print(f"  Kernel  : {TRAIN_KERNEL}")

    print("  " + "="*55 + "\n")

    logger.info(f"Pipeline started | TRAIN_KERNEL={TRAIN_KERNEL}")

    # ---- Load Data ----
    X_train, y_train, X_test, y_test = load_all()

    # ---- Decide which kernels to run ----
    if TRAIN_KERNEL == "all":
        kernels_to_run = ALL_KERNELS
    elif TRAIN_KERNEL in ALL_KERNELS:
        kernels_to_run = [TRAIN_KERNEL]
    else:
        raise ValueError(
            f"Unknown TRAIN_KERNEL value: '{TRAIN_KERNEL}'\n"
            f"Set it to 'all' or one of: {ALL_KERNELS}"
        )

    # ---- Run each kernel ----
    all_metrics = []

    for kernel in kernels_to_run:
        metrics = run_one_kernel(kernel, X_train, y_train, X_test, y_test)
        all_metrics.append(metrics)

    # ---- Final Summary ----
    total_elapsed = time.time() - total_start
    minutes = int(total_elapsed // 60)
    seconds = int(total_elapsed % 60)

    print("\n  " + "="*55)
    print("  PIPELINE COMPLETE")
    print("  " + "="*55)
    print(f"  Kernels trained  : {[m['kernel'] for m in all_metrics]}")
    print(f"  Total time       : {minutes}m {seconds}s")
    print(f"  Results saved in : results/")
    print(f"  Log file         : logs/pipeline.log")
    print("  " + "="*55)

    # ---- Quick comparison table ----
    if len(all_metrics) > 1:
        divider = "-" * 75
        print(f"\n  Quick Comparison")
        print(f"  {divider}")
        print(f"  {'KERNEL':<12} {'ACCURACY':>10} {'F1':>10} {'ROC-AUC':>10} {'MCC':>10}")
        print(f"  {divider}")
        for m in all_metrics:
            print(
                f"  {m['kernel']:<12} "
                f"{m['accuracy']:>10.4f} "
                f"{m['f1_weighted']:>10.4f} "
                f"{m['roc_auc']:>10.4f} "
                f"{m['mcc']:>10.4f}"
            )
        print(f"  {divider}\n")

    logger.info(f"Pipeline complete | total time: {total_elapsed:.1f}s | kernels: {[m['kernel'] for m in all_metrics]}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
