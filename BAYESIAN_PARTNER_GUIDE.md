# Bayesian Optimization -- Partner Documentation

## Project: SVM Sentiment Classification (IMDB)
## Author of Base Code: Group 4
## This Document: Guide for Bayesian Optimization Integration

---

## 1. What the Existing Code Already Does For You

The base codebase is fully modular. You do not need to rewrite anything.
Every piece you need is already built and tested. Your job is to write
one new file: bayesian_search.py. Everything else plugs in directly.

Here is what each existing file gives you:

### paths.py
Defines every directory path in the project. Import this and you
automatically know where data lives, where to save results, and where
logs go. If you are running on your own machine, just change BASE_DIR
in this file and nothing else needs to change.

### config.py
Gives you ALL_KERNELS (the list of all four kernels), LOG_FORMAT,
and LOG_DATE_FMT. Import these directly -- do not redefine them.

### search_config.py
This is the most important file for you. It already defines:
- SEARCH_KERNEL: set this to "all" or a specific kernel name
- CV_FOLDS: 5 (stratified)
- CV_SCORING: "f1_macro" (this is the score you must optimize)
- CV_RANDOM_STATE: 42
- RANDOM_SEARCH_DISTRIBUTIONS: the parameter distributions per kernel

The distributions in search_config.py are a good starting point for
your Bayesian search spaces too. You can widen or narrow them as needed.

### data_loader.py
Call load_data("train") to get X_train and y_train.
Call load_data("test") to get X_test and y_test.
Never touch test data during search. Only use it after retraining.

### trainer.py
After Bayesian search finds the best parameters, call run_training()
to retrain the final model on the full training set and save it.
It handles the heartbeat, timestamps, folder creation, and saving
model.pkl, hyperparameters.json, and metadata.json automatically.

### evaluator.py
After retraining, call evaluate_and_save() to compute and save all
metrics: Accuracy, Precision, Recall, F1, MCC, ROC-AUC, PR-AUC,
training time, inference time, support vectors, y_test, y_pred,
y_scores, and the classification report. You do not need to write
any of this yourself.

---

## 2. What You Need to Build

You need to write one file:

    bayesian_search.py

And optionally add Bayesian-specific settings to:

    search_config.py  (just add a new section at the bottom)

That is all.

---

## 3. Recommended Library

Use scikit-optimize (skopt). It integrates cleanly with sklearn and
is the most straightforward choice for this project.

Install:
    pip install scikit-optimize

The key function you will use is:
    skopt.gp_minimize()

Or the sklearn-compatible wrapper:
    skopt.BayesSearchCV()

For this project, BayesSearchCV is the cleaner approach because it
mirrors the cross_validate pattern already used in grid_search.py
and random_search.py.

---

## 4. What to Add in search_config.py

Add this section at the bottom of search_config.py:

    from skopt.space import Real, Integer, Categorical

    BAYESIAN_SEARCH_SPACES = {

        "linear": [
            Real(0.01, 100, prior="log-uniform", name="C"),
        ],

        "rbf": [
            Real(0.01, 100, prior="log-uniform", name="C"),
            Categorical(["scale", "auto", 0.001, 0.01, 0.1], name="gamma"),
        ],

        "poly": [
            Real(0.01, 100, prior="log-uniform", name="C"),
            Integer(2, 5, name="degree"),
            Categorical(["scale", "auto"], name="gamma"),
            Real(0.0, 2.0, prior="uniform", name="coef0"),
        ],

        "sigmoid": [
            Real(0.01, 100, prior="log-uniform", name="C"),
            Categorical(["scale", "auto", 0.001, 0.01, 0.1], name="gamma"),
            Real(-1.0, 1.0, prior="uniform", name="coef0"),
        ],
    }

    BAYESIAN_N_CALLS = {
        "linear"  : 20,
        "rbf"     : 25,
        "poly"    : 30,
        "sigmoid" : 25,
    }

    BAYESIAN_N_INITIAL_POINTS = 5

Note: n_initial_points is the number of random exploration steps before
the Gaussian Process starts guiding the search. 5 is a reasonable default.

---

## 5. Output Folder Structure

Follow the exact same pattern as grid_search.py and random_search.py.
This is critical for comparison later.

    results/search/bayesian/svm_rbf_bayesian_2025-04-15_1430/
        search_metadata.json
        cv_results.csv
        best_params.json
        convergence_log.json      <-- THIS IS UNIQUE TO BAYESIAN

The convergence_log.json is what makes Bayesian different from the others.
It must record, for each evaluation step:
    - evaluation number (1, 2, 3, ... up to n_calls)
    - parameters tried at that step
    - CV score at that step
    - best score seen so far up to that step

This data is what produces the convergence curve plot later.
Without it, you cannot show how Bayesian search improves over iterations.

---

## 6. What cv_results.csv Must Contain

Follow the exact same column structure as grid_search.py and random_search.py
so that comparison scripts can read all three files the same way.

Columns:
    iteration
    [one column per parameter, e.g. C, gamma, degree, coef0]
    mean_test_score
    std_test_score
    rank
    mean_fit_time
    std_fit_time
    mean_score_time
    std_score_time
    total_cv_time
    fold_scores

The fold_scores column should be a string representation of the list,
exactly as done in grid_search.py and random_search.py.

---

## 7. What search_metadata.json Must Contain

    {
        "method"              : "bayesian_search",
        "kernel"              : "rbf",
        "timestamp"           : "2025-04-15_1430",
        "cv_folds"            : 5,
        "cv_scoring"          : "f1_macro",
        "n_calls"             : 25,
        "n_initial_points"    : 5,
        "search_space"        : { ... description of space ... },
        "num_candidates"      : 25,
        "total_search_time"   : 3421.5,
        "best_params"         : { "C": 4.72, "gamma": "scale" },
        "best_cv_score"       : 0.9134
    }

---

## 8. What convergence_log.json Must Contain

This is unique to Bayesian and does not exist for grid or random search.
It records how the best score improved (or did not) at each step.

    [
        {
            "evaluation"     : 1,
            "params_tried"   : { "C": 0.15, "gamma": "auto" },
            "score_this_step": 0.8821,
            "best_so_far"    : 0.8821
        },
        {
            "evaluation"     : 2,
            "params_tried"   : { "C": 3.7, "gamma": "scale" },
            "score_this_step": 0.9012,
            "best_so_far"    : 0.9012
        },
        ...
    ]

This file is what produces the convergence curve plot, which is one of
the most important visuals in the final report. It shows that Bayesian
search finds good parameters faster than random or grid search.

---

## 9. Step-by-Step Flow for bayesian_search.py

Follow this exact flow:

Step 1 -- Import from existing files
    from paths         import RESULTS_DIR, LOGS_DIR, LOG_FILE
    from data_loader   import load_data
    from search_config import (SEARCH_KERNEL, CV_FOLDS, CV_SCORING,
                               CV_RANDOM_STATE, BAYESIAN_SEARCH_SPACES,
                               BAYESIAN_N_CALLS, BAYESIAN_N_INITIAL_POINTS,
                               SEARCH_RESULTS_SUBDIR)
    from config        import ALL_KERNELS, LOG_FORMAT, LOG_DATE_FMT

Step 2 -- Load training data only
    X_train, y_train = load_data("train")

Step 3 -- For each kernel to search:
    a. Create timestamped output folder:
       results/search/bayesian/svm_{kernel}_bayesian_{timestamp}/

    b. Define the objective function:
       The objective takes a parameter set, builds an SVC, runs
       cross_validate with StratifiedKFold(n_splits=CV_FOLDS),
       and returns the NEGATIVE mean CV score (gp_minimize minimizes,
       so negate the score to maximize it).
       Also append to the convergence log inside the objective function
       so every evaluation is recorded.

    c. Call gp_minimize() or BayesSearchCV with n_calls and
       n_initial_points from search_config.

    d. Collect all results from the optimizer's result object.

    e. Save cv_results.csv, search_metadata.json, best_params.json,
       and convergence_log.json.

Step 4 -- Print a live summary after each evaluation (same style as
    grid_search.py and random_search.py).

Step 5 -- Print a final summary table at the end showing best params
    and best CV score per kernel.

---

## 10. After Search is Done -- Retraining

After bayesian_search.py finishes, retraining uses the exact same
pipeline.py that is already written. You only need to:

1. Open config.py
2. Find the kernel you want to retrain
3. Update its DEFAULT_PARAMS with the best_params found by Bayesian search
4. Set TRAIN_KERNEL to that kernel (or "all")
5. Run: python pipeline.py

The pipeline will train the model, evaluate it, and save everything
in the standard results folder with a timestamp. No changes to pipeline.py
are needed.

---

## 11. Scoring -- Important

Use f1_macro consistently. Do not change this.
Grid search uses f1_macro. Random search uses f1_macro.
Bayesian must also use f1_macro.

If you use a different score, the three methods cannot be compared fairly.
The CV_SCORING variable in search_config.py is already set to "f1_macro".
Always read from there. Never hardcode it inside bayesian_search.py.

---

## 12. Quick Checklist Before You Start

    [ ] pip install scikit-optimize
    [ ] Read paths.py to understand where files will be saved
    [ ] Read search_config.py fully before writing anything
    [ ] Read grid_search.py as a reference for structure and style
    [ ] Add BAYESIAN_SEARCH_SPACES and BAYESIAN_N_CALLS to search_config.py
    [ ] Make sure your cv_results.csv columns match grid and random exactly
    [ ] Make sure convergence_log.json is saved -- it is the most unique output
    [ ] Never use test data during search
    [ ] Always use CV_SCORING from search_config, never hardcode f1_macro

---

## 13. Files You Will Touch

    WRITE:   bayesian_search.py       (new file, your main task)
    EDIT:    search_config.py         (add Bayesian spaces and n_calls)
    READ:    paths.py                 (for folder paths)
    READ:    config.py                (for ALL_KERNELS and log format)
    READ:    data_loader.py           (to load train data)
    READ:    grid_search.py           (as your structural reference)
    IGNORE:  trainer.py               (used only after search, by pipeline.py)
    IGNORE:  evaluator.py             (used only after search, by pipeline.py)
    IGNORE:  pipeline.py              (used only after search, not during it)
    IGNORE:  random_search.py         (reference only, no edits needed)

---

## 14. Output Summary

When your script finishes running for one kernel, the partner
running pipeline.py should see a folder like:

    results/search/bayesian/svm_rbf_bayesian_2025-04-15_1430/
        search_metadata.json      <-- run summary
        cv_results.csv            <-- all 25 evaluations with scores
        best_params.json          <-- just the best params, clean and simple
        convergence_log.json      <-- step by step best score progression

That is four files per kernel, per run. Every run is timestamped
and never overwrites a previous run.
