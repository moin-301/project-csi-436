# =============================================================================
# bayesian_search.py
#
# PURPOSE : Run Bayesian Optimization for one or all SVM kernels.
#           Uses Gaussian Process to intelligently explore the search space.
#           Saves every trial result in the same format as grid and random search.
#           Does NOT retrain or evaluate on test data.
#           Use pipeline.py for final retraining after search is done.
#
# HOW TO RUN:
#   python bayesian_search.py
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
#   results/search/bayesian/svm_rbf_bayesian_2025-04-15_1430/
#     search_metadata.json   -- run summary and best params
#     cv_results.csv         -- one row per iteration (same format as grid/random)
#     best_params.json       -- just the best parameters
#     convergence_log.json   -- score progression per iteration (Bayesian only)
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
from skopt import gp_minimize

sys.path.insert(0, str(Path(__file__).parent))

from paths import RESULTS_DIR, LOGS_DIR, LOG_FILE
from data_loader import load_data
from search_config import (
    SEARCH_KERNEL,
    BAYESIAN_SEARCH_SPACES,
    BAYESIAN_N_CALLS,
    BAYESIAN_N_INITIAL_POINTS,
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
logger = logging.getLogger("bayesian_search")


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
    Create and return the output folder for this Bayesian search run.
    Format: results/search/bayesian/svm_{kernel}_bayesian_{timestamp}/
    """
    timestamp   = datetime.now().strftime("%Y-%m-%d_%H%M")
    folder_name = f"svm_{kernel}_bayesian_{timestamp}"
    run_folder  = RESULTS_DIR / SEARCH_RESULTS_SUBDIR / "bayesian" / folder_name
    run_folder.mkdir(parents=True, exist_ok=True)
    return run_folder, timestamp


# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================

def make_objective(kernel, space, X_train, y_train, all_trials, convergence_log):
    """
    Returns a closure that gp_minimize calls on every iteration.
    Captures all CV results needed for fair comparison with grid and random search.

    Parameters
    ----------
    kernel         : str
    space          : list of skopt dimension objects
    X_train        : sparse matrix
    y_train        : numpy array
    all_trials     : list -- appended to on every call
    convergence_log: list -- appended to on every call

    Returns
    -------
    objective : callable -- takes a list of param values, returns negative score
    """

    cv = StratifiedKFold(
        n_splits     = CV_FOLDS,
        shuffle      = CV_SHUFFLE,
        random_state = CV_RANDOM_STATE,
    )

    def objective(params):
        # Map list of values back to named dict using dimension names
        param_dict = {dim.name: val for dim, val in zip(space, params)}

        model = SVC(
            kernel       = kernel,
            probability  = False,
            random_state = CV_RANDOM_STATE,
            **param_dict,
        )

        trial_start = time.time()

        cv_output = cross_validate(
            model,
            X_train,
            y_train,
            cv                 = cv,
            scoring            = CV_SCORING,
            return_train_score = False,
            n_jobs             = -1,
        )

        total_cv_time = time.time() - trial_start

        # -- All scores needed for fair comparison --
        fold_scores     = cv_output["test_score"]
        mean_score      = float(np.mean(fold_scores))
        std_score       = float(np.std(fold_scores))
        fit_times       = cv_output["fit_time"]
        score_times     = cv_output["score_time"]
        mean_fit_time   = float(np.mean(fit_times))
        std_fit_time    = float(np.std(fit_times))
        mean_score_time = float(np.mean(score_times))
        std_score_time  = float(np.std(score_times))

        iteration = len(all_trials) + 1

        # -- Best so far for convergence tracking --
        best_so_far = max(
            [t["mean_test_score"] for t in all_trials] + [mean_score]
        )
        best_params_so_far = param_dict.copy()
        if all_trials:
            best_trial = max(all_trials, key=lambda t: t["mean_test_score"])
            if best_trial["mean_test_score"] >= mean_score:
                best_params_so_far = best_trial["params"]

        # -- Trial record (matches cv_results.csv column format) --
        trial = {
            "kernel"          : kernel,
            "method"          : "bayesian_search",
            "iteration"       : iteration,
            "params"          : param_dict,
            "C"               : param_dict.get("C"),
            "degree"          : param_dict.get("degree"),
            "gamma"           : param_dict.get("gamma"),
            "coef0"           : param_dict.get("coef0"),
            "mean_test_score" : round(mean_score, 6),
            "std_test_score"  : round(std_score, 6),
            "fold_1_score"    : round(float(fold_scores[0]), 6),
            "fold_2_score"    : round(float(fold_scores[1]), 6),
            "fold_3_score"    : round(float(fold_scores[2]), 6),
            "fold_4_score"    : round(float(fold_scores[3]), 6),
            "fold_5_score"    : round(float(fold_scores[4]), 6),
            "mean_fit_time"   : round(mean_fit_time, 4),
            "std_fit_time"    : round(std_fit_time, 4),
            "mean_score_time" : round(mean_score_time, 4),
            "std_score_time"  : round(std_score_time, 4),
            "total_cv_time"   : round(total_cv_time, 2),
            # Bayesian-only fields
            "best_score_so_far" : round(best_so_far, 6),
            "best_params_so_far": str(best_params_so_far),
        }

        all_trials.append(trial)

        # -- Convergence log entry --
        convergence_log.append({
            "evaluation"      : iteration,
            "params_tried"    : param_dict,
            "score_this_step" : round(mean_score, 6),
            "best_so_far"     : round(best_so_far, 6),
        })

        marker = "  <-- NEW BEST" if mean_score >= best_so_far else ""
        print(
            f"  Iter {iteration:>3}  |  score: {mean_score:.4f}  "
            f"(best: {best_so_far:.4f})  "
            f"[{total_cv_time:.1f}s]"
            f"{marker}"
        )

        logger.info(
            f"Iter {iteration} | {param_dict} | "
            f"score={mean_score:.4f} | best={best_so_far:.4f}"
        )

        return -mean_score  # Negate because gp_minimize minimizes

    return objective


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(
    run_folder     : Path,
    kernel         : str,
    timestamp      : str,
    space          : list,
    all_trials     : list,
    best_params    : dict,
    best_score     : float,
    total_time     : float,
    convergence_log: list,
):
    """
    Save all results in the same format as grid_search.py and random_search.py,
    plus convergence_log.json which is unique to Bayesian search.

    Files created:
      search_metadata.json  -- run summary
      cv_results.csv        -- one row per iteration
      best_params.json      -- best parameters only
      convergence_log.json  -- score progression per iteration
    """

    num_iterations = len(all_trials)

    # -- Assign rank --
    sorted_scores = sorted(
        [t["mean_test_score"] for t in all_trials],
        reverse=True,
    )
    for trial in all_trials:
        trial["rank"]    = sorted_scores.index(trial["mean_test_score"]) + 1
        trial["is_best"] = trial["mean_test_score"] == best_score

    # ------------------------------------------------------------------
    # search_metadata.json
    # ------------------------------------------------------------------
    search_space_desc = {}
    for dim in space:
        search_space_desc[dim.name] = str(dim)

    metadata = {
        "method"             : "bayesian_search",
        "kernel"             : kernel,
        "timestamp"          : timestamp,
        "cv_folds"           : CV_FOLDS,
        "cv_scoring"         : CV_SCORING,
        "n_calls"            : BAYESIAN_N_CALLS[kernel],
        "n_initial_points"   : BAYESIAN_N_INITIAL_POINTS,
        "search_space"       : search_space_desc,
        "num_iterations"     : num_iterations,
        "total_search_time"  : round(total_time, 2),
        "best_params"        : best_params,
        "best_cv_score"      : round(best_score, 6),
    }

    meta_path = run_folder / "search_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Metadata saved   : {meta_path}")
    print(f"  Saved: search_metadata.json")

    # ------------------------------------------------------------------
    # cv_results.csv
    # Same column structure as grid and random search for consolidation
    # ------------------------------------------------------------------
    fieldnames = [
        "kernel", "method", "timestamp", "iteration", "rank", "is_best",
        "C", "degree", "gamma", "coef0",
        "mean_test_score", "std_test_score",
        "fold_1_score", "fold_2_score", "fold_3_score", "fold_4_score", "fold_5_score",
        "mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time",
        "total_cv_time", "best_cv_score",
        # Bayesian-only columns appended at end
        "best_score_so_far", "best_params_so_far",
    ]

    csv_path = run_folder / "cv_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        sorted_trials = sorted(all_trials, key=lambda x: x["rank"])

        for trial in sorted_trials:
            row = {
                "kernel"             : trial["kernel"],
                "method"             : trial["method"],
                "timestamp"          : timestamp,
                "iteration"          : trial["iteration"],
                "rank"               : trial["rank"],
                "is_best"            : trial["is_best"],
                "C"                  : trial["C"],
                "degree"             : trial["degree"],
                "gamma"              : trial["gamma"],
                "coef0"              : trial["coef0"],
                "mean_test_score"    : trial["mean_test_score"],
                "std_test_score"     : trial["std_test_score"],
                "fold_1_score"       : trial["fold_1_score"],
                "fold_2_score"       : trial["fold_2_score"],
                "fold_3_score"       : trial["fold_3_score"],
                "fold_4_score"       : trial["fold_4_score"],
                "fold_5_score"       : trial["fold_5_score"],
                "mean_fit_time"      : trial["mean_fit_time"],
                "std_fit_time"       : trial["std_fit_time"],
                "mean_score_time"    : trial["mean_score_time"],
                "std_score_time"     : trial["std_score_time"],
                "total_cv_time"      : trial["total_cv_time"],
                "best_cv_score"      : round(best_score, 6),
                "best_score_so_far"  : trial["best_score_so_far"],
                "best_params_so_far" : trial["best_params_so_far"],
            }
            writer.writerow(row)

    logger.info(f"CV results saved : {csv_path}")
    print(f"  Saved: cv_results.csv  ({num_iterations} rows)")

    # ------------------------------------------------------------------
    # best_params.json
    # ------------------------------------------------------------------
    best_path = run_folder / "best_params.json"
    with open(best_path, "w") as f:
        json.dump({"kernel": kernel, **best_params}, f, indent=4)
    logger.info(f"Best params saved: {best_path}")
    print(f"  Saved: best_params.json")

    # ------------------------------------------------------------------
    # convergence_log.json  (Bayesian only)
    # ------------------------------------------------------------------
    conv_path = run_folder / "convergence_log.json"
    with open(conv_path, "w") as f:
        json.dump(convergence_log, f, indent=4)
    logger.info(f"Convergence log  : {conv_path}")
    print(f"  Saved: convergence_log.json")


# =============================================================================
# RUN BAYESIAN SEARCH FOR ONE KERNEL
# =============================================================================

def run_bayesian_search_one_kernel(kernel: str, X_train, y_train):
    """
    Run Bayesian Optimization for a single kernel.

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

    space      = BAYESIAN_SEARCH_SPACES[kernel]
    n_calls    = BAYESIAN_N_CALLS[kernel]
    all_trials     = []
    convergence_log = []

    run_folder, timestamp = make_run_folder(kernel)

    # ---- Print search plan ----
    print(f"\n  {'='*60}")
    print(f"  BAYESIAN SEARCH  |  Kernel: {kernel.upper()}")
    print(f"  {'='*60}")
    print(f"  n_calls          : {n_calls}")
    print(f"  n_initial_points : {BAYESIAN_N_INITIAL_POINTS}")
    print(f"  CV folds         : {CV_FOLDS}  (Stratified)")
    print(f"  Scoring metric   : {CV_SCORING}")
    print(f"  Output folder    : {run_folder}")
    print(f"  {'='*60}\n")

    logger.info(
        f"Bayesian Search | kernel={kernel} | "
        f"n_calls={n_calls} | n_initial_points={BAYESIAN_N_INITIAL_POINTS}"
    )

    # ---- Build objective ----
    objective = make_objective(
        kernel         = kernel,
        space          = space,
        X_train        = X_train,
        y_train        = y_train,
        all_trials     = all_trials,
        convergence_log = convergence_log,
    )

    # ---- Run optimizer ----
    total_start = time.time()

    res = gp_minimize(
        func             = objective,
        dimensions       = space,
        n_calls          = n_calls,
        n_initial_points = BAYESIAN_N_INITIAL_POINTS,
        random_state     = CV_RANDOM_STATE,
        verbose          = False,
    )

    total_time = time.time() - total_start

    # ---- Extract best result ----
    best_trial  = max(all_trials, key=lambda t: t["mean_test_score"])
    best_score  = best_trial["mean_test_score"]
    best_params = best_trial["params"].copy()

    # ---- Print summary ----
    print(f"\n  {'='*60}")
    print(f"  Bayesian Search Complete  |  Kernel: {kernel.upper()}")
    print(f"  Best params    : {best_params}")
    print(f"  Best CV {CV_SCORING}: {best_score:.4f}")
    print(f"  Total time     : {total_time:.1f}s")
    print(f"  {'='*60}\n")

    logger.info(
        f"Bayesian Search done | kernel={kernel} | "
        f"best_params={best_params} | best_score={best_score:.4f}"
    )

    # ---- Save everything ----
    save_results(
        run_folder      = run_folder,
        kernel          = kernel,
        timestamp       = timestamp,
        space           = space,
        all_trials      = all_trials,
        best_params     = best_params,
        best_score      = best_score,
        total_time      = total_time,
        convergence_log = convergence_log,
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
    print("  SVM -- BAYESIAN SEARCH")
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

    logger.info(f"Bayesian Search pipeline started | SEARCH_KERNEL={SEARCH_KERNEL}")

    # Load training data only -- test data is never touched during search
    print("Loading training data...")
    X_train, y_train = load_data("train")
    print("Training data loaded.\n")

    # Run search for each kernel
    summary = []

    for kernel in kernels_to_search:
        best_params, best_score, run_folder = run_bayesian_search_one_kernel(
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
    print("  BAYESIAN SEARCH COMPLETE -- SUMMARY")
    print("  " + "="*60)
    print(f"  {'KERNEL':<12}  {'BEST CV SCORE':>15}  BEST PARAMS")
    print(f"  {'-'*55}")
    for s in summary:
        print(f"  {s['kernel']:<12}  {s['best_score']:>15.4f}  {s['best_params']}")
    print(f"\n  Total time : {minutes}m {seconds}s")
    print(f"  Results    : results/search/bayesian/")
    print("  " + "="*60 + "\n")

    logger.info(f"Bayesian Search pipeline complete | total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
