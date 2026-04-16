# SVM Sentiment Classification -- Project Documentation
## Group 4: Irfan, Siddique, White-Jenkins, Wojtalewski


# 1. PROJECT OVERVIEW
The pipeline supports four kernels: linear, RBF, polynomial, and sigmoid. It is designed to run in three phases:

    Phase 1 -- Baseline training with default parameters
    Phase 2 -- Hyperparameter search (Grid Search and Random Search)
    Phase 3 -- Bayesian Optimization (separate, handled by partner)

Every run saves all outputs with a human-readable timestamp so nothing
is ever overwritten and everything is identifiable later.

---

# 2. FOLDER STRUCTURE

## What the project looks like on disk

    Project/Codes/
            |--src/
            |   |-- config.py                  Training settings and kernel parameters
            |    -- paths.py                   All file and folder paths
            |    -- search_config.py           All hyperparameter search settings
            |
            |    -- data_loader.py             Loads .npz and .npy data files
            |    -- trainer.py                 Builds, trains, and saves SVM models
            |    -- evaluator.py               Computes and saves all evaluation metrics
            |    -- pipeline.py                Entry point -- the only file you run for training
            |
            |    -- grid_search.py             Runs exhaustive Grid Search
            |    -- random_search.py           Runs Randomized Search
            |
            |-- data/
            |   |-- processed/
            |       |-- train_tfidf.npz    Sparse TF-IDF matrix (training)
            |       |-- train_labels.npy   Labels for training (0 or 1)
            |       |-- test_tfidf.npz     Sparse TF-IDF matrix (test)
            |       |-- test_labels.npy    Labels for test (0 or 1)
            |
            |-- results/
            |   |-- svm_linear_2025-04-15_1430/       One folder per training run
            |   |   |-- model.pkl
            |   |   |-- hyperparameters.json
            |   |   |-- metadata.json
            |   |   |-- metrics.json
            |   |   |-- classification_report.txt
            |   |   |-- y_test.npy
            |   |   |-- y_pred.npy
            |   |   |-- y_scores.npy
            |   |
            |   |-- search/
            |       |-- grid/
            |       |   |-- svm_rbf_grid_2025-04-15_1430/
            |       |       |-- search_metadata.json
            |       |       |-- cv_results.csv
            |       |       |-- best_params.json
            |       |
            |       |-- random/
            |           |-- svm_rbf_random_2025-04-15_1431/
            |               |-- search_metadata.json
            |               |-- cv_results.csv
            |               |-- best_params.json
            |
            |-- logs/
                |-- pipeline.log           Full log of every run


## Key design rules

- Every training run creates its own timestamped subfolder inside results/
- Timestamp format: svm_{kernel}_{YYYY-MM-DD}_{HHMM}
- Old results are never overwritten -- every run is independent
- Search results live inside results/search/ and are separate from training results

---

# 3. FILE DOCUMENTATION

---

## paths.py

PURPOSE:
    The only file that defines where things are on disk.
    If you clone the project on a new machine, this is the only
    file you may need to edit. Everything else reads from here.

WHAT IT DEFINES:
    BASE_DIR      -- project root, default is current directory (.)
    DATA_DIR      -- where .npz and .npy data files are expected
    TRAIN_MATRIX  -- full path to train_tfidf.npz
    TRAIN_LABELS  -- full path to train_labels.npy
    TEST_MATRIX   -- full path to test_tfidf.npz
    TEST_LABELS   -- full path to test_labels.npy
    RESULTS_DIR   -- where all training outputs go
    LOGS_DIR      -- where log files go
    LOG_FILE      -- full path to pipeline.log

WHEN TO EDIT:
    Only if your data folder is somewhere other than data/processed/
    relative to the project root. If you follow the standard folder
    structure, you never need to touch this file.

IMPORTED BY:
    data_loader.py, trainer.py, evaluator.py, pipeline.py,
    grid_search.py, random_search.py

---

## config.py

PURPOSE:
    Controls which kernels get trained and with what parameters.
    This is the file you edit before running pipeline.py.

WHAT IT DEFINES:

    TRAIN_KERNEL
        Set this to control what pipeline.py trains.
        "all"     -- trains all four kernels one by one
        "linear"  -- trains only linear
        "rbf"     -- trains only RBF
        "poly"    -- trains only polynomial
        "sigmoid" -- trains only sigmoid

    ALL_KERNELS
        Master list of all kernels: ["linear", "rbf", "poly", "sigmoid"]
        To add a new kernel, add its name here.

    DEFAULT_PARAMS
        Dictionary of default hyperparameters per kernel.
        These are used during baseline training (Phase 1).
        To change a parameter, edit the value here.
        Example:
            "rbf": {"C": 1.0, "gamma": "scale"}

    HEARTBEAT_INTERVAL
        How often (in seconds) the training progress message prints.
        Default is 30 seconds.

    LOG_FORMAT, LOG_DATE_FMT
        Shared logging format used by all modules.

WHEN TO EDIT:
    Before every pipeline.py run. Set TRAIN_KERNEL and DEFAULT_PARAMS
    as needed. After hyperparameter search, update DEFAULT_PARAMS with
    the best parameters found and retrain.

IMPORTED BY:
    pipeline.py, trainer.py, grid_search.py, random_search.py

---

## search_config.py

PURPOSE:
    Controls all hyperparameter search settings for Grid Search
    and Random Search. Neither search script hardcodes anything --
    everything comes from here.

WHAT IT DEFINES:

    SEARCH_KERNEL
        Same concept as TRAIN_KERNEL in config.py but for search.
        "all"    -- searches all kernels
        "linear" -- searches only linear
        (and so on for rbf, poly, sigmoid)

    CV_FOLDS
        Number of cross-validation folds. Set to 5 (Stratified 5-fold).
        Do not change this unless you have a specific reason, because
        grid and random search must use the same folds to compare fairly.

    CV_SCORING
        The metric used to pick the best hyperparameters.
        Set to "f1_macro". Must stay the same across all search methods
        including Bayesian optimization so comparison is fair.

    CV_RANDOM_STATE
        Random seed for reproducibility. Set to 42.

    RANDOM_SEARCH_N_ITER
        Dictionary defining how many random configurations to try
        per kernel. Values are chosen to be close to the number of
        grid search combinations for each kernel so comparison is fair.

    GRID_SEARCH_SPACES
        Dictionary defining the exact parameter values to try per kernel.
        Grid search tries every combination from these lists.
        Example for RBF:
            "C": [0.1, 1.0, 10.0, 100.0]
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1]

    RANDOM_SEARCH_DISTRIBUTIONS
        Dictionary defining parameter distributions per kernel.
        Random search samples from these distributions.
        loguniform -- for C and gamma (equal probability on log scale)
        uniform    -- for coef0 (equal probability on linear scale)
        list       -- discrete choices picked randomly

    SEARCH_RESULTS_SUBDIR
        Name of the subfolder inside results/ where search outputs go.
        Default is "search".

WHEN TO EDIT:
    Set SEARCH_KERNEL before running grid_search.py or random_search.py.
    Adjust search spaces or distributions if you want to explore different
    parameter ranges. Do not change CV_FOLDS or CV_SCORING.

IMPORTED BY:
    grid_search.py, random_search.py

---

## data_loader.py

PURPOSE:
    Loads preprocessed data files from disk.
    Does one job only -- reading data.
    Does not clean, transform, train, or evaluate anything.

FUNCTIONS:

    load_data(split)
        Loads the TF-IDF matrix and labels for one split.
        split must be "train" or "test".
        Returns X (sparse matrix) and y (numpy array of 0s and 1s).
        Raises FileNotFoundError if the data files are missing.
        Prints shape and class balance to console.

    load_all()
        Convenience wrapper that calls load_data("train") and
        load_data("test") and returns all four variables:
        X_train, y_train, X_test, y_test

WHAT IT READS:
    data/processed/train_tfidf.npz
    data/processed/train_labels.npy
    data/processed/test_tfidf.npz
    data/processed/test_labels.npy

IMPORTED BY:
    pipeline.py, grid_search.py, random_search.py

---

## trainer.py

PURPOSE:
    Builds, trains, and saves SVM models.
    Does one job only -- training and saving.
    Does not evaluate or plot anything.

CLASSES:

    Heartbeat
        A background thread that prints a live status line every N seconds
        while model.fit() is running. This prevents the terminal from
        looking frozen during long training runs (RBF and poly can take
        30+ minutes). Automatically starts before fit() and stops after,
        even if training crashes midway.

FUNCTIONS:

    make_timestamp()
        Returns the current time as a string in format: 2025-04-15_1430

    make_run_folder(kernel)
        Creates and returns the timestamped output folder for a run.
        Format: results/svm_{kernel}_{timestamp}/

    build_model(kernel, params)
        Creates an untrained SVC object with the given kernel and params.
        Sets probability=False (uses decision_function for ROC/PR curves,
        which is faster than probability=True).
        Sets random_state=42 for reproducibility.

    train_model(model, X_train, y_train)
        Trains the model with a live heartbeat running in the background.
        Returns the trained model and elapsed time in seconds.

    save_run(model, kernel, params, elapsed, X_train, run_folder, timestamp)
        Saves three files into run_folder:
            model.pkl            -- trained SVC using joblib
            hyperparameters.json -- exact params used
            metadata.json        -- kernel, timestamp, timing, shapes,
                                    number of support vectors, random state
        Returns metadata dict.

    run_training(kernel, X_train, y_train, params)
        The main entry point. Calls make_run_folder, build_model,
        train_model, and save_run in sequence.
        Returns model, metadata dict, and run_folder path.
        If params is None, reads from DEFAULT_PARAMS in config.py.

IMPORTED BY:
    pipeline.py

---

## evaluator.py

PURPOSE:
    Evaluates a trained model on test data and saves every output needed
    for later plotting and comparison tables.
    Does one job only -- measuring and saving results.
    Does not train, search, or plot anything.

METRICS COMPUTED:

    Core metrics (weighted averages):
        Accuracy, Precision, Recall, F1, MCC

    Per-class metrics:
        Precision, Recall, F1 for each class (Negative and Positive)

    Advanced metrics:
        ROC-AUC   -- uses decision_function scores (not probabilities)
        PR-AUC    -- same

    Efficiency:
        Inference time in seconds
        Number of support vectors

    Structural:
        Full classification report (text)
        Confusion matrix

FILES SAVED (inside run_folder):
    metrics.json               -- all numeric scores as a dictionary
    classification_report.txt  -- full per-class text breakdown
    y_test.npy                 -- true labels (needed for all plots)
    y_pred.npy                 -- predicted labels (for confusion matrix)
    y_scores.npy               -- decision function scores (for ROC and PR curves)

FUNCTIONS:

    evaluate_and_save(model, X_test, y_test, run_folder, metadata)
        Main entry point. Runs all evaluation steps, prints results to
        console, saves all files, and returns the metrics dictionary.

    _print_metrics(metrics, report_text)
        Prints a formatted summary to the console including confusion matrix
        and full classification report.

    _save_all(run_folder, metrics, report_text, y_test, y_pred, y_scores)
        Saves all five output files into run_folder.

IMPORTED BY:
    pipeline.py

---

## pipeline.py

PURPOSE:
    The only file you need to run for training.
    Reads settings from config.py, loads data, trains, evaluates,
    and saves everything. Does not search or tune hyperparameters.

HOW TO RUN:
    python pipeline.py

WHAT IT DOES:
    1. Reads TRAIN_KERNEL from config.py
    2. Sets up logging to both terminal and logs/pipeline.log
    3. Loads train and test data via data_loader.py
    4. For each kernel to train:
       a. Calls run_training() from trainer.py
       b. Calls evaluate_and_save() from evaluator.py
       c. All outputs saved automatically in timestamped folder
    5. Prints a quick comparison table if more than one kernel is trained

DOES NOT TOUCH:
    grid_search.py, random_search.py, search_config.py

OUTPUT:
    results/svm_{kernel}_{timestamp}/    one folder per kernel trained
    logs/pipeline.log                    full session log

---

## grid_search.py

PURPOSE:
    Runs exhaustive Grid Search for one or all kernels.
    Tries every combination in GRID_SEARCH_SPACES from search_config.py.
    Does not retrain on test data. Use pipeline.py after search.

HOW TO RUN:
    python grid_search.py

WHAT IT DOES:
    1. Reads SEARCH_KERNEL from search_config.py
    2. Loads training data only (test data never touched during search)
    3. For each kernel:
       a. Builds full list of parameter combinations using ParameterGrid
       b. For each combination, runs StratifiedKFold cross-validation
       c. Prints live progress: combo number, params, CV score, timing
       d. Tracks best score and best params seen so far
       e. After all combinations, assigns ranks and saves results

FILES SAVED per kernel run:
    results/search/grid/svm_{kernel}_grid_{timestamp}/
        search_metadata.json  -- method, kernel, folds, scoring, best params,
                                  best score, total time, search space
        cv_results.csv        -- one row per combination tried, columns:
                                  params, mean_test_score, std_test_score,
                                  rank, mean_fit_time, std_fit_time,
                                  mean_score_time, std_score_time,
                                  total_cv_time, fold_scores
        best_params.json      -- just the best parameters, clean and simple

USE cv_results.csv FOR:
    Heatmaps of C vs gamma for RBF
    Score vs C plots for linear
    Degree vs score plots for polynomial
    Runtime vs score plots

---

## random_search.py

PURPOSE:
    Runs Randomized Search for one or all kernels.
    Samples from distributions in RANDOM_SEARCH_DISTRIBUTIONS from
    search_config.py. Faster than grid search for large parameter spaces.
    Does not retrain on test data. Use pipeline.py after search.

HOW TO RUN:
    python random_search.py

WHAT IT DOES:
    1. Reads SEARCH_KERNEL from search_config.py
    2. Loads training data only
    3. For each kernel:
       a. Samples n_iter random configurations from distributions
       b. For each configuration, runs StratifiedKFold cross-validation
       c. Prints live progress: iteration number, sampled params, CV score
       d. Tracks best score and best params seen so far
       e. After all iterations, assigns ranks and saves results

FILES SAVED per kernel run:
    results/search/random/svm_{kernel}_random_{timestamp}/
        search_metadata.json  -- same structure as grid search metadata
                                  plus n_iter and distributions description
        cv_results.csv        -- same columns as grid search plus
                                  iteration column showing which sample
                                  each row corresponds to
        best_params.json      -- just the best parameters

KEY DIFFERENCE FROM GRID SEARCH:
    Uses loguniform distributions for C and gamma so the search explores
    a much wider range than grid search in the same number of evaluations.
    n_iter per kernel is set in RANDOM_SEARCH_N_ITER in search_config.py
    and is kept close to the number of grid combinations for fair comparison.

---

# 4. ENVIRONMENT SETUP

## Step 1 -- Make sure Python is installed

    python --version

You need Python 3.8 or higher. On M1 Mac, use the native ARM version.

## Step 2 -- Create a virtual environment

    python -m venv svm_env

## Step 3 -- Activate the environment

On Mac or Linux:
    source svm_env/bin/activate

On Windows:
    svm_env\Scripts\activate

You should see (svm_env) at the start of your terminal line.

## Step 4 -- Install all required packages

    pip install numpy scipy scikit-learn joblib matplotlib seaborn

That is everything the project needs. No other packages required
for the training and search phase.

For Bayesian optimization (partner's work), one extra package:
    pip install scikit-optimize

## Step 5 -- Verify installation

    python -c "import numpy, scipy, sklearn, joblib; print('All good')"

If you see "All good" the environment is ready.

## Full list of packages and what they are used for

    numpy          -- array operations, saving y_test/y_pred/y_scores
    scipy          -- loading .npz sparse matrices, loguniform distribution
    scikit-learn   -- SVC, cross-validation, all metrics
    joblib         -- saving and loading trained model files (.pkl)
    matplotlib     -- plotting (used later in comparison phase)
    seaborn        -- confusion matrix heatmaps (used later)

---

# 5. HOW TO RUN THE PROJECT

## Phase 1 -- Baseline Training

1. Open config.py
2. Set TRAIN_KERNEL to "all" or a specific kernel
3. Check DEFAULT_PARAMS to confirm the parameter values
4. Run:
       python pipeline.py
5. Results are saved in results/svm_{kernel}_{timestamp}/

## Phase 2 -- Hyperparameter Search

For Grid Search:
1. Open search_config.py
2. Set SEARCH_KERNEL to "all" or a specific kernel
3. Adjust GRID_SEARCH_SPACES if needed
4. Run:
       python grid_search.py
5. Results are saved in results/search/grid/

For Random Search:
1. Open search_config.py
2. Set SEARCH_KERNEL to "all" or a specific kernel
3. Adjust RANDOM_SEARCH_N_ITER if needed
4. Run:
       python random_search.py
5. Results are saved in results/search/random/

## Phase 3 -- Retrain with Best Parameters

1. Open the best_params.json from your search results
2. Copy those values into DEFAULT_PARAMS in config.py for that kernel
3. Set TRAIN_KERNEL to that kernel
4. Run:
       python pipeline.py
5. New results saved with new timestamp

---

## What someone needs to do to run the code after cloning

Step 1 -- Clone the repository
    git clone https://github.com/your-repo-name/project.git
    cd project

Step 2 -- Create and activate environment
    python -m venv svm_env
    source svm_env/bin/activate

Step 3 -- Install packages
    pip install numpy scipy scikit-learn joblib matplotlib seaborn

Step 4 -- Confirm data files are present
    The folder data/processed/ should already contain:
        train_tfidf.npz
        train_labels.npy
        test_tfidf.npz
        test_labels.npy
    If not, run the preprocessing script first.

Step 5 -- Run the pipeline
    python pipeline.py

No path changes needed. paths.py uses BASE_DIR = Path(".")
which means it always looks relative to wherever you run the script from.
As long as you run python pipeline.py from inside the project folder,
all paths resolve correctly automatically.

## If the .npz files are too large for GitHub -- use Git LFS

Step 1 -- Install Git LFS
    brew install git-lfs       (on Mac)
    git lfs install

Step 2 -- Track large files
    git lfs track "*.npz"
    git lfs track "*.npy"
    git add .gitattributes

Step 3 -- Add and commit normally
    git add .
    git commit -m "Add processed data and source files"
    git push

When someone clones the repo, Git LFS downloads the large files
automatically if they have Git LFS installed.
