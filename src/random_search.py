# =============================================================================
# random_search.py
#
# PURPOSE : Run Randomized Search for one or all SVM kernels.
#           Saves every trial result, not just the best one.
#           Does NOT retrain or evaluate on test data.
#           Use pipeline.py for final retraining after search is done.
#
# HOW TO RUN:
#   python random_search.py
#
# TO SEARCH ONE KERNEL:
#   Open search_config.py and set:
#     SEARCH_KERNEL = "rbf"
#
# TO SEARCH ALL KERNELS:
#   Open search_config.py and set:
#     SEARCH_KERNEL = "all"
#
# WHY RANDOM SEARCH:
#   Grid Search tries every combination, which is expensive.
#   Random Search samples from distributions, covering a wider range
#   in fewer iterations. With n_iter close to grid combinations,
#   the comparison between the two methods stays fair.
#
# OUTPUT STRUCTURE:
#   results/search/random/svm_rbf_random_2025-04-15_1430/
#     search_metadata.json   -- run summary and best params
#     cv_results.csv         -- one row per sampled configuration tried
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
from sklearn.model_selection import StratifiedKFold, cross_validate

sys.path.insert(0, str(Path(__file__).parent))

from paths         import RESULTS_DIR, LOGS_DIR, LOG_FILE
from data_loader   import load_data
from search_config import (
    SEARCH_KERNEL,
    RANDOM_SEARCH_DISTRIBUTIONS,
    RANDOM_SEARCH_N_ITER,
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
logger = logging.getLogger("random_search")


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
    Create and return the output folder for this random search run.
    Format: results/search/random/svm_{kernel}_random_{timestamp}/
    """
    timestamp   = datetime.now().strftime("%Y-%m-%d_%H%M")
    folder_name = f"svm_{kernel}_random_{timestamp}"
    run_folder  = RESULTS_DIR / SEARCH_RESULTS_SUBDIR / "random" / folder_name
    run_folder.mkdir(parents=True, exist_ok=True)
    return run_folder, timestamp


# =============================================================================
# SAMPLE ONE CONFIGURATION
# =============================================================================

def sample_params(distributions: dict, rng: np.random.RandomState):
    """
    Sample one set of hyperparameters from the given distributions.

    For each parameter:
      - If it is a list     : pick one value randomly
      - If it is a scipy rv : call .rvs() to draw one sample

    Parameters
    ----------
    distributions : dict  -- from RANDOM_SEARCH_DISTRIBUTIONS in search_config.py
    rng           : numpy RandomState -- for reproducible sampling from lists

    Returns
    -------
    params : dict -- one sampled parameter set
    """

    params = {}

    for key, dist in distributions.items():
        if isinstance(dist, list):
            # Discrete list -- pick one value randomly
            params[key] = dist[rng.randint(len(dist))]
        else:
            # Continuous scipy distribution -- draw one sample
            value = dist.rvs(random_state=rng)
            # Round floats for cleaner display and saving
            if isinstance(value, float):
                value = round(float(value), 6)
            params[key] = value

    return params


# =============================================================================
# EVALUATE ONE SAMPLED CONFIGURATION
# =============================================================================

def evaluate_one_config(params: dict, kernel: str, X_train, y_train, cv):
    """
    Run cross-validation for one sampled parameter configuration.

    Parameters
    ----------
    params  : dict
    kernel  : str
    X_train : sparse matrix
    y_train : numpy array
    cv      : StratifiedKFold object

    Returns
    -------
    result : dict -- mean score, std, fit times, score times, fold scores
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
        cv                 = cv,
        scoring            = CV_SCORING,
        return_train_score = False,
        n_jobs             = -1,
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
    distributions : dict,
    n_iter        : int,
    all_trials    : list,
    best_params   : dict,
    best_score    : float,
    total_time    : float,
):
    """
    Save all search results into the run folder.

    Files created:
      search_metadata.json  -- summary of the full run
      cv_results.csv        -- one row per sampled configuration
      best_params.json      -- just the best parameters
    """

    num_trials = len(all_trials)

    # Collect all param keys seen across all trials for CSV columns
    all_param_keys = set()
    for trial in all_trials:
        all_param_keys.update(trial["params"].keys())
    param_keys = sorted(all_param_keys)

    # ------------------------------------------------------------------
    # search_metadata.json
    # ------------------------------------------------------------------
    dist_description = {}
    for k, v in distributions.items():
        if isinstance(v, list):
            dist_description[k] = v
        else:
            dist_description[k] = str(v)

    metadata = {
        "method"              : "random_search",
        "kernel"              : kernel,
        "timestamp"           : timestamp,
        "cv_folds"            : CV_FOLDS,
        "cv_scoring"          : CV_SCORING,
        "n_iter"              : n_iter,
        "distributions"       : dist_description,
        "num_trials"          : num_trials,
        "total_search_time"   : round(total_time, 2),
        "best_params"         : best_params,
        "best_cv_score"       : round(best_score, 6),
    }

    meta_path = run_folder / "search_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Metadata saved   : {meta_path}")
    print(f"  Saved: search_metadata.json")

    # ------------------------------------------------------------------
    # cv_results.csv
    # One row per sampled configuration.
    # Use this later for scatter plots, sensitivity analysis, etc.
    # ------------------------------------------------------------------

    fieldnames = (
        ["iteration"]
        + param_keys
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

        sorted_trials = sorted(all_trials, key=lambda x: x["rank"])

        for trial in sorted_trials:
            row = {"iteration": trial["iteration"]}
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
    print(f"  Saved: cv_results.csv  ({num_trials} rows)")

    # ------------------------------------------------------------------
    # best_params.json
    # ------------------------------------------------------------------
    best_path = run_folder / "best_params.json"
    with open(best_path, "w") as f:
        json.dump({"kernel": kernel, **best_params}, f, indent=4)
    logger.info(f"Best params saved: {best_path}")
    print(f"  Saved: best_params.json")


# =============================================================================
# RUN RANDOM SEARCH FOR ONE KERNEL
# =============================================================================

def run_random_search_one_kernel(kernel: str, X_train, y_train):
    """
    Run Random Search for a single kernel.

    Parameters
    ----------
    kernel  : str
    X_train : sparse matrix
    y_train : numpy array

    Returns
    -------
    best_params : dict
    best_score  : float
    run_folder  : Path
    """

    distributions = RANDOM_SEARCH_DISTRIBUTIONS[kernel]
    n_iter        = RANDOM_SEARCH_N_ITER[kernel]
    total_fits    = n_iter * CV_FOLDS

    run_folder, timestamp = make_run_folder(kernel)

    # ---- Print search plan ----
    print(f"\n  {'='*60}")
    print(f"  RANDOM SEARCH  |  Kernel: {kernel.upper()}")
    print(f"  {'='*60}")
    print(f"  Iterations       : {n_iter}")
    print(f"  CV folds         : {CV_FOLDS}  (Stratified)")
    print(f"  Total model fits : {total_fits}")
    print(f"  Scoring metric   : {CV_SCORING}")
    print(f"  Output folder    : {run_folder}")
    print(f"  {'='*60}\n")

    logger.info(
        f"Random Search | kernel={kernel} | n_iter={n_iter} | "
        f"folds={CV_FOLDS} | total_fits={total_fits}"
    )

    # ---- Cross-validation object ----
    cv = StratifiedKFold(
        n_splits     = CV_FOLDS,
        shuffle      = CV_SHUFFLE,
        random_state = CV_RANDOM_STATE,
    )

    rng = np.random.RandomState(CV_RANDOM_STATE)

    # ---- Sample and evaluate ----
    all_trials  = []
    best_score  = -1.0
    best_params = None

    total_start = time.time()

    for i in range(1, n_iter + 1):

        params = sample_params(distributions, rng)

        print(f"  Iter {i:>3}/{n_iter}  |  {params}")

        scores = evaluate_one_config(params, kernel, X_train, y_train, cv)

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
            f"Iter {i}/{n_iter} | {params} | "
            f"score={mean_score:.4f} | std={std_score:.4f}"
        )

        if is_best:
            best_score  = mean_score
            best_params = params.copy()

        trial = {
            "iteration" : i,
            "params"    : params,
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
    print(f"  Random Search Complete  |  Kernel: {kernel.upper()}")
    print(f"  Best params      : {best_params}")
    print(f"  Best CV {CV_SCORING}  : {best_score:.4f}")
    print(f"  Total time       : {total_time:.1f}s")
    print(f"  {'='*60}\n")

    logger.info(
        f"Random Search done | kernel={kernel} | "
        f"best_params={best_params} | best_score={best_score:.4f}"
    )

    # ---- Save everything ----
    save_results(
        run_folder    = run_folder,
        kernel        = kernel,
        timestamp     = timestamp,
        distributions = distributions,
        n_iter        = n_iter,
        all_trials    = all_trials,
        best_params   = best_params,
        best_score    = best_score,
        total_time    = total_time,
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
    print("  SVM -- RANDOM SEARCH")
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

    logger.info(f"Random Search pipeline started | SEARCH_KERNEL={SEARCH_KERNEL}")

    # Load training data only -- test data is not touched during search
    print("Loading training data...")
    X_train, y_train = load_data("train")
    print("Training data loaded.\n")

    # Run search for each kernel
    summary = []

    for kernel in kernels_to_search:
        best_params, best_score, run_folder = run_random_search_one_kernel(
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
    print("  RANDOM SEARCH COMPLETE -- SUMMARY")
    print("  " + "="*60)
    print(f"  {'KERNEL':<12}  {'BEST CV SCORE':>15}  BEST PARAMS")
    print(f"  {'-'*55}")
    for s in summary:
        print(f"  {s['kernel']:<12}  {s['best_score']:>15.4f}  {s['best_params']}")
    print(f"\n  Total time : {minutes}m {seconds}s")
    print(f"  Results    : results/search/random/")
    print("  " + "="*60 + "\n")

    logger.info(f"Random Search pipeline complete | total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
