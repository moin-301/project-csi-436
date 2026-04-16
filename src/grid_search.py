# =============================================================================
# grid_search.py
#
# PURPOSE : Run exhaustive Grid Search for one or all SVM kernels.
#           Saves every trial result, not just the best one.
#           Does NOT retrain or evaluate on test data.
#           Use pipeline.py for final retraining after search is done.
#
# HOW TO RUN:
#   python grid_search.py
#
# TO SEARCH ONE KERNEL:
#   Open search_config.py and set:
#     SEARCH_KERNEL = "rbf"
#
# TO SEARCH ALL KERNELS:
#   Open search_config.py and set:
#     SEARCH_KERNEL = "all"
#
# OUTPUT STRUCTURE:
#   results/search/grid/svm_rbf_grid_2025-04-15_1430/
#     search_metadata.json   -- run summary and best params
#     cv_results.csv         -- one row per parameter combination tried
#     best_params.json       -- just the best parameters (easy to load later)
# =============================================================================

import sys
import csv
import json
import time
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_validate

sys.path.insert(0, str(Path(__file__).parent))

from paths         import RESULTS_DIR, LOGS_DIR, LOG_FILE
from data_loader   import load_data
from search_config import (
    SEARCH_KERNEL,
    GRID_SEARCH_SPACES,
    CV_FOLDS,
    CV_SCORING,
    CV_RANDOM_STATE,
    CV_SHUFFLE,
    SEARCH_RESULTS_SUBDIR,
)
from config import (
    ALL_KERNELS,
    LOG_FORMAT,
    LOG_DATE_FMT,
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("grid_search")


# =============================================================================
# SETUP LOGGING
# =============================================================================

def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level    = logging.INFO,
        format   = LOG_FORMAT,
        datefmt  = LOG_DATE_FMT,
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE),
        ],
    )


# =============================================================================
# FOLDER HELPER
# =============================================================================

def make_run_folder(kernel: str):
    """
    Create and return the output folder for this grid search run.
    Format: results/search/grid/svm_{kernel}_grid_{timestamp}/
    """
    timestamp   = datetime.now().strftime("%Y-%m-%d_%H%M")
    folder_name = f"svm_{kernel}_grid_{timestamp}"
    run_folder  = RESULTS_DIR / SEARCH_RESULTS_SUBDIR / "grid" / folder_name
    run_folder.mkdir(parents=True, exist_ok=True)
    return run_folder, timestamp


# =============================================================================
# EVALUATE ONE PARAMETER COMBINATION
# =============================================================================

def evaluate_one_combo(params: dict, kernel: str, X_train, y_train, cv):
    """
    Run cross-validation for one parameter combination and return all scores.

    Parameters
    ----------
    params  : dict -- one specific set of hyperparameters
    kernel  : str  -- kernel name
    X_train : sparse matrix
    y_train : numpy array
    cv      : StratifiedKFold object

    Returns
    -------
    result : dict -- mean score, std, fit times, score times, per-fold scores
    """

    model = SVC(
        kernel       = kernel,
        probability  = False,
        random_state = 42,
        **params
    )

    start = time.time()

    cv_output = cross_validate(
        model,
        X_train,
        y_train,
        cv              = cv,
        scoring         = CV_SCORING,
        return_train_score = False,
        n_jobs          = -1,
    )

    elapsed = time.time() - start

    result = {
        "mean_test_score" : float(np.mean(cv_output["test_score"])),
        "std_test_score"  : float(np.std(cv_output["test_score"])),
        "fold_scores"     : [round(float(s), 6) for s in cv_output["test_score"]],
        "mean_fit_time"   : float(np.mean(cv_output["fit_time"])),
        "std_fit_time"    : float(np.std(cv_output["fit_time"])),
        "mean_score_time" : float(np.mean(cv_output["score_time"])),
        "std_score_time"  : float(np.std(cv_output["score_time"])),
        "total_cv_time"   : round(elapsed, 2),
    }

    return result


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(
    run_folder    : Path,
    kernel        : str,
    timestamp     : str,
    search_space  : dict,
    all_trials    : list,
    best_params   : dict,
    best_score    : float,
    total_time    : float,
):
    """
    Save all search results into the run folder.

    Files created:
      search_metadata.json  -- summary of the full run
      cv_results.csv        -- one row per combination tried
      best_params.json      -- just the best parameters
    """

    num_candidates = len(all_trials)

    # ------------------------------------------------------------------
    # search_metadata.json
    # ------------------------------------------------------------------
    metadata = {
        "method"            : "grid_search",
        "kernel"            : kernel,
        "timestamp"         : timestamp,
        "cv_folds"          : CV_FOLDS,
        "cv_scoring"        : CV_SCORING,
        "search_space"      : {k: (v if not hasattr(v, 'rvs') else str(v)) for k, v in search_space.items()},
        "num_candidates"    : num_candidates,
        "total_search_time" : round(total_time, 2),
        "best_params"       : best_params,
        "best_cv_score"     : round(best_score, 6),
    }

    meta_path = run_folder / "search_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Metadata saved   : {meta_path}")
    print(f"  Saved: search_metadata.json")

    # ------------------------------------------------------------------
    # cv_results.csv
    # One row per parameter combination tried.
    # This is the file you will use for heatmaps and sensitivity plots.
    # ------------------------------------------------------------------

    # Build column names dynamically from param keys
    param_keys = list(search_space.keys())

    fieldnames = (
        param_keys
        + [
            "mean_test_score",
            "std_test_score",
            "rank",
            "mean_fit_time",
            "std_fit_time",
            "mean_score_time",
            "std_score_time",
            "total_cv_time",
            "fold_scores",
        ]
    )

    csv_path = run_folder / "cv_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Sort by rank before writing
        sorted_trials = sorted(all_trials, key=lambda x: x["rank"])

        for trial in sorted_trials:
            row = {}
            for k in param_keys:
                row[k] = trial["params"].get(k, "")
            row["mean_test_score"] = round(trial["mean_test_score"], 6)
            row["std_test_score"]  = round(trial["std_test_score"],  6)
            row["rank"]            = trial["rank"]
            row["mean_fit_time"]   = round(trial["mean_fit_time"],   4)
            row["std_fit_time"]    = round(trial["std_fit_time"],     4)
            row["mean_score_time"] = round(trial["mean_score_time"],  4)
            row["std_score_time"]  = round(trial["std_score_time"],   4)
            row["total_cv_time"]   = trial["total_cv_time"]
            row["fold_scores"]     = str(trial["fold_scores"])
            writer.writerow(row)

    logger.info(f"CV results saved : {csv_path}")
    print(f"  Saved: cv_results.csv  ({num_candidates} rows)")

    # ------------------------------------------------------------------
    # best_params.json
    # ------------------------------------------------------------------
    best_path = run_folder / "best_params.json"
    with open(best_path, "w") as f:
        json.dump({"kernel": kernel, **best_params}, f, indent=4)
    logger.info(f"Best params saved: {best_path}")
    print(f"  Saved: best_params.json")


# =============================================================================
# RUN GRID SEARCH FOR ONE KERNEL
# =============================================================================

def run_grid_search_one_kernel(kernel: str, X_train, y_train):
    """
    Run Grid Search for a single kernel.

    Parameters
    ----------
    kernel  : str
    X_train : sparse matrix
    y_train : numpy array

    Returns
    -------
    best_params : dict  -- best hyperparameters found
    best_score  : float -- best mean CV score
    run_folder  : Path  -- where results were saved
    """

    search_space   = GRID_SEARCH_SPACES[kernel]
    all_combos     = list(ParameterGrid(search_space))
    total_combos   = len(all_combos)
    total_fits     = total_combos * CV_FOLDS

    run_folder, timestamp = make_run_folder(kernel)

    # ---- Print search plan ----
    print(f"\n  {'='*60}")
    print(f"  GRID SEARCH  |  Kernel: {kernel.upper()}")
    print(f"  {'='*60}")
    print(f"  Search space     : {search_space}")
    print(f"  Combinations     : {total_combos}")
    print(f"  CV folds         : {CV_FOLDS}  (Stratified)")
    print(f"  Total model fits : {total_fits}")
    print(f"  Scoring metric   : {CV_SCORING}")
    print(f"  Output folder    : {run_folder}")
    print(f"  {'='*60}\n")

    logger.info(
        f"Grid Search | kernel={kernel} | combos={total_combos} | "
        f"folds={CV_FOLDS} | total_fits={total_fits}"
    )

    # ---- Cross-validation object ----
    cv = StratifiedKFold(
        n_splits     = CV_FOLDS,
        shuffle      = CV_SHUFFLE,
        random_state = CV_RANDOM_STATE,
    )

    # ---- Try every combination ----
    all_trials = []
    best_score = -1.0
    best_params = None

    total_start = time.time()

    for i, params in enumerate(all_combos, start=1):

        print(f"  Combo {i:>3}/{total_combos}  |  {params}")

        scores = evaluate_one_combo(params, kernel, X_train, y_train, cv)

        mean_score = scores["mean_test_score"]
        std_score  = scores["std_test_score"]

        is_best = mean_score > best_score
        marker  = "  <-- NEW BEST" if is_best else ""

        print(
            f"             CV {CV_SCORING}: {mean_score:.4f}  "
            f"(+/- {std_score:.4f})  "
            f"[{scores['total_cv_time']}s]"
            f"{marker}"
        )

        logger.info(
            f"Combo {i}/{total_combos} | {params} | "
            f"score={mean_score:.4f} | std={std_score:.4f}"
        )

        if is_best:
            best_score  = mean_score
            best_params = params.copy()

        trial = {
            "params"         : params,
            **scores,
        }
        all_trials.append(trial)

    total_time = time.time() - total_start

    # ---- Assign ranks ----
    sorted_scores = sorted(
        [t["mean_test_score"] for t in all_trials],
        reverse=True
    )
    for trial in all_trials:
        trial["rank"] = sorted_scores.index(trial["mean_test_score"]) + 1

    # ---- Print summary ----
    print(f"\n  {'='*60}")
    print(f"  Grid Search Complete  |  Kernel: {kernel.upper()}")
    print(f"  Best params      : {best_params}")
    print(f"  Best CV {CV_SCORING}  : {best_score:.4f}")
    print(f"  Total time       : {total_time:.1f}s")
    print(f"  {'='*60}\n")

    logger.info(
        f"Grid Search done | kernel={kernel} | "
        f"best_params={best_params} | best_score={best_score:.4f}"
    )

    # ---- Save everything ----
    save_results(
        run_folder   = run_folder,
        kernel       = kernel,
        timestamp    = timestamp,
        search_space = search_space,
        all_trials   = all_trials,
        best_params  = best_params,
        best_score   = best_score,
        total_time   = total_time,
    )

    return best_params, best_score, run_folder


# =============================================================================
# MAIN
# =============================================================================

def main():

    setup_logging()

    total_start = time.time()

    print("\n")
    print("  " + "="*60)
    print("  SVM -- GRID SEARCH")
    print("  " + "="*60)

    if SEARCH_KERNEL == "all":
        kernels_to_search = ALL_KERNELS
        print(f"  Mode    : Search ALL kernels")
        print(f"  Kernels : {kernels_to_search}")
    elif SEARCH_KERNEL in ALL_KERNELS:
        kernels_to_search = [SEARCH_KERNEL]
        print(f"  Mode    : Search ONE kernel")
        print(f"  Kernel  : {SEARCH_KERNEL}")
    else:
        raise ValueError(
            f"Unknown SEARCH_KERNEL: '{SEARCH_KERNEL}'\n"
            f"Set it to 'all' or one of: {ALL_KERNELS}"
        )

    print("  " + "="*60 + "\n")

    logger.info(f"Grid Search pipeline started | SEARCH_KERNEL={SEARCH_KERNEL}")

    # Load training data only -- test data is not touched during search
    print("Loading training data...")
    X_train, y_train = load_data("train")
    print("Training data loaded.\n")

    # Run search for each kernel
    summary = []

    for kernel in kernels_to_search:
        best_params, best_score, run_folder = run_grid_search_one_kernel(
            kernel  = kernel,
            X_train = X_train,
            y_train = y_train,
        )
        summary.append({
            "kernel"      : kernel,
            "best_params" : best_params,
            "best_score"  : best_score,
            "folder"      : str(run_folder),
        })

    # Final summary table
    total_time = time.time() - total_start
    minutes    = int(total_time // 60)
    seconds    = int(total_time % 60)

    print("\n  " + "="*60)
    print("  GRID SEARCH COMPLETE -- SUMMARY")
    print("  " + "="*60)
    print(f"  {'KERNEL':<12}  {'BEST CV SCORE':>15}  BEST PARAMS")
    print(f"  {'-'*55}")
    for s in summary:
        print(f"  {s['kernel']:<12}  {s['best_score']:>15.4f}  {s['best_params']}")
    print(f"\n  Total time : {minutes}m {seconds}s")
    print(f"  Results    : results/search/grid/")
    print("  " + "="*60 + "\n")

    logger.info(f"Grid Search pipeline complete | total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
