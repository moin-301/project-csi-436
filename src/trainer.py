# =============================================================================
# trainer.py
#
# PURPOSE : Build, train, and save SVM models.
#           This module does ONE job only -- training and saving.
#           It does NOT evaluate or plot anything.
#
# WHAT IT SAVES (inside results/svm_{kernel}_{timestamp}/):
#   model.pkl            -- the trained model (joblib format)
#   hyperparameters.json -- exact parameter values used
#   metadata.json        -- kernel, timestamp, timing, shape, support vectors
# =============================================================================

import time
import json
import logging
import threading
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.svm import SVC

from config import HEARTBEAT_INTERVAL, DEFAULT_PARAMS
from paths  import RESULTS_DIR

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("trainer")


# =============================================================================
# TIMESTAMP HELPER
# =============================================================================

def make_timestamp():
    """
    Return a human-readable timestamp string.
    Format: 2025-04-15_1430
    """
    return datetime.now().strftime("%Y-%m-%d_%H%M")


def make_run_folder(kernel: str):
    """
    Create and return the output folder for this training run.
    Folder name format: svm_{kernel}_{timestamp}
    Example: results/svm_rbf_2025-04-15_1430/
    """
    timestamp  = make_timestamp()
    folder_name = f"svm_{kernel}_{timestamp}"
    run_folder  = RESULTS_DIR / folder_name
    run_folder.mkdir(parents=True, exist_ok=True)
    return run_folder, timestamp


# =============================================================================
# HEARTBEAT -- Background Thread
# =============================================================================

class Heartbeat:
    """
    Prints a live status line every N seconds while training runs.
    Lets you know the program has not crashed or frozen.

    Usage:
        hb = Heartbeat("SVC (rbf)")
        hb.start()
        model.fit(X_train, y_train)
        hb.stop()
    """

    def __init__(self, model_label: str, interval: int = HEARTBEAT_INTERVAL):
        self.model_label = model_label
        self.interval    = interval
        self._stop_flag  = threading.Event()
        self._thread     = threading.Thread(target=self._run, daemon=True)
        self._start_time = None

    def _run(self):
        while not self._stop_flag.wait(timeout=self.interval):
            elapsed = time.time() - self._start_time
            print(f"  Still training {self.model_label}... {elapsed:.0f}s elapsed", flush=True)
            logger.info(f"Heartbeat | {self.model_label} | {elapsed:.0f}s elapsed")

    def start(self):
        self._start_time = time.time()
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        self._thread.join()


# =============================================================================
# BUILD MODEL
# =============================================================================

def build_model(kernel: str, params: dict):
    """
    Create an untrained SVC with the given kernel and parameters.

    Parameters
    ----------
    kernel : str  -- "linear", "rbf", "poly", or "sigmoid"
    params : dict -- hyperparameter values from config or manual override

    Returns
    -------
    model : untrained SVC object
    """

    logger.info(f"Building SVC | kernel={kernel} | params={params}")

    model = SVC(
        kernel      = kernel,
        probability = False,   # Using decision_function for ROC/PR -- faster
        random_state= 42,
        **params
    )

    return model


# =============================================================================
# TRAIN
# =============================================================================

def train_model(model, X_train, y_train):
    """
    Train a model with a live heartbeat. Returns the trained model and
    the elapsed time in seconds.

    Parameters
    ----------
    model   : untrained SVC
    X_train : sparse TF-IDF matrix
    y_train : numpy label array

    Returns
    -------
    model        : trained SVC
    elapsed_time : float -- seconds taken to train
    """

    label = type(model).__name__
    if hasattr(model, "kernel"):
        label += f" ({model.kernel})"

    print(f"\n  {'='*55}")
    print(f"  Training : {label}")
    print(f"  Samples  : {X_train.shape[0]:,}")
    print(f"  Features : {X_train.shape[1]:,}")
    print(f"  Heartbeat every {HEARTBEAT_INTERVAL}s")
    print(f"  {'='*55}")

    logger.info(f"Training {label} | samples={X_train.shape[0]} | features={X_train.shape[1]}")

    heartbeat = Heartbeat(model_label=label)
    heartbeat.start()

    start = time.time()
    try:
        model.fit(X_train, y_train)
    finally:
        heartbeat.stop()

    elapsed = time.time() - start

    print(f"\n  Training done. Time: {elapsed:.1f}s")
    print(f"  {'='*55}\n")
    logger.info(f"Training complete | {label} | {elapsed:.1f}s")

    return model, elapsed


# =============================================================================
# SAVE
# =============================================================================

def save_run(model, kernel: str, params: dict, elapsed: float, X_train, run_folder: Path, timestamp: str):
    """
    Save the trained model and all metadata for this run.

    Files created inside run_folder:
      model.pkl            -- trained SVC (joblib)
      hyperparameters.json -- exact params used
      metadata.json        -- all run info

    Parameters
    ----------
    model      : trained SVC
    kernel     : str   -- kernel name
    params     : dict  -- hyperparameters used
    elapsed    : float -- training time in seconds
    X_train    : sparse matrix -- used to record feature count and sample count
    run_folder : Path  -- where to save everything
    timestamp  : str   -- the timestamp string for this run
    """

    # -- Model --
    model_path = run_folder / "model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved      : {model_path}")

    # -- Hyperparameters --
    params_path = run_folder / "hyperparameters.json"
    with open(params_path, "w") as f:
        json.dump({"kernel": kernel, **params}, f, indent=4)
    logger.info(f"Hyperparams saved: {params_path}")

    # -- Metadata --
    num_support_vectors = int(model.n_support_.sum()) if hasattr(model, "n_support_") else None

    metadata = {
        "run_folder"           : str(run_folder),
        "timestamp"            : timestamp,
        "kernel"               : kernel,
        "hyperparameters"      : {"kernel": kernel, **params},
        "training_time_seconds": round(elapsed, 2),
        "train_samples"        : X_train.shape[0],
        "num_features"         : X_train.shape[1],
        "num_support_vectors"  : num_support_vectors,
        "random_state"         : 42,
    }

    meta_path = run_folder / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Metadata saved   : {meta_path}")

    print(f"  Run folder       : {run_folder}")
    print(f"  Model saved      : model.pkl")
    print(f"  Hyperparams saved: hyperparameters.json")
    print(f"  Metadata saved   : metadata.json")

    return metadata


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_training(kernel: str, X_train, y_train, params: dict = None):
    """
    Full training pipeline for one kernel:
      1. Create output folder with timestamp
      2. Build model
      3. Train with heartbeat
      4. Save model + hyperparameters + metadata

    Parameters
    ----------
    kernel  : str        -- "linear", "rbf", "poly", or "sigmoid"
    X_train : sparse matrix
    y_train : numpy array
    params  : dict (optional) -- override default params from config

    Returns
    -------
    model      : trained SVC
    metadata   : dict -- all info about this run
    run_folder : Path -- where everything was saved

    Example
    -------
    model, metadata, run_folder = run_training("rbf", X_train, y_train)
    """

    # Use default params from config if none provided
    if params is None:
        params = DEFAULT_PARAMS.get(kernel, {})

    # Create timestamped output folder
    run_folder, timestamp = make_run_folder(kernel)

    print(f"\n  Run folder: {run_folder}")

    # Build and train
    model          = build_model(kernel, params)
    model, elapsed = train_model(model, X_train, y_train)

    # Save everything
    metadata = save_run(
        model      = model,
        kernel     = kernel,
        params     = params,
        elapsed    = elapsed,
        X_train    = X_train,
        run_folder = run_folder,
        timestamp  = timestamp,
    )

    return model, metadata, run_folder
