# =============================================================================
# config.py
#
# PURPOSE : All training settings in one place.
#           Control what kernels to train and with what parameters.
#
# HOW TO USE:
#   - To train ONE kernel    : set TRAIN_KERNEL = "rbf"
#   - To train ALL kernels   : set TRAIN_KERNEL = "all"
#
# HOW TO ADD A NEW KERNEL:
#   1. Add its name to ALL_KERNELS list
#   2. Add its default parameters to DEFAULT_PARAMS
#   That is all. Nothing else in the project needs to change.
# =============================================================================

# ---------------------------------------------------------------------------
# TRAINING MODE
#
# TRAIN_KERNEL controls what the pipeline trains when you run pipeline.py
#
# Options:
#   "all"      -- trains every kernel in ALL_KERNELS one by one
#   "linear"   -- trains only the linear kernel
#   "rbf"      -- trains only the rbf kernel
#   "poly"     -- trains only the poly kernel
#   "sigmoid"  -- trains only the sigmoid kernel
# ---------------------------------------------------------------------------

TRAIN_KERNEL = "all"

# ---------------------------------------------------------------------------
# ALL KERNELS
# This is the master list. Add a new kernel name here to include it.
# ---------------------------------------------------------------------------

ALL_KERNELS = ["linear", "rbf", "poly", "sigmoid"]

# ---------------------------------------------------------------------------
# DEFAULT HYPERPARAMETERS PER KERNEL
#
# These are the values used during baseline training (no search).
# Every parameter is written out explicitly so nothing is hidden.
#
# To change a parameter for one kernel, just edit the value here.
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "linear"  : {
        "C"      : 1.0,
    },
    "rbf"     : {
        "C"      : 1.0,
        "gamma"  : "scale",
    },
    "poly"    : {
        "C"      : 1.0,
        "degree" : 3,
        "gamma"  : "scale",
        "coef0"  : 0.0,
    },
    "sigmoid" : {
        "C"      : 1.0,
        "gamma"  : "scale",
        "coef0"  : 0.0,
    },
}

# ---------------------------------------------------------------------------
# HEARTBEAT
# Prints a live status message every N seconds during training
# so you know the program has not crashed.
# ---------------------------------------------------------------------------

HEARTBEAT_INTERVAL = 30

# ---------------------------------------------------------------------------
# LOGGING FORMAT
# Shared across all modules so every log line looks the same.
# ---------------------------------------------------------------------------

LOG_FORMAT   = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATE_FMT = "%Y-%m-%d %H:%M:%S"
