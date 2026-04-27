# =============================================================================
# search_config.py
#
# PURPOSE : All hyperparameter search settings in one place.
#           Grid Search, Random Search, and Bayesian Search all read from here.
#           Nothing is hardcoded inside the search scripts.
#
# HOW TO USE:
#   - To search ONE kernel  : set SEARCH_KERNEL = "rbf"
#   - To search ALL kernels : set SEARCH_KERNEL = "all"
#
# HOW TO ADD A NEW KERNEL:
#   1. Add its grid to GRID_SEARCH_SPACES
#   2. Add its distributions to RANDOM_SEARCH_DISTRIBUTIONS
#   3. Add its space to BAYESIAN_SEARCH_SPACES
#   4. Add its iteration counts to RANDOM_SEARCH_N_ITER and BAYESIAN_N_CALLS
#   That is all. The search scripts do not need to change.
# =============================================================================

from scipy.stats import loguniform, uniform

# ---------------------------------------------------------------------------
# SEARCH MODE
#
# SEARCH_KERNEL controls which kernel(s) get tuned.
#
# Options:
#   "all"      -- searches every kernel defined below
#   "linear"   -- searches only linear
#   "rbf"      -- searches only rbf
#   "poly"     -- searches only poly
#   "sigmoid"  -- searches only sigmoid
# ---------------------------------------------------------------------------

SEARCH_KERNEL = "linear"

# ---------------------------------------------------------------------------
# CROSS-VALIDATION SETTINGS
# Same settings used by Grid Search, Random Search, and Bayesian Search
# so the comparison between all three methods stays fair.
# ---------------------------------------------------------------------------

CV_FOLDS        = 5               # Stratified 5-fold
CV_SCORING      = "f1_macro"      # Main tuning score -- same for all methods
CV_RANDOM_STATE = 31              # For reproducibility
CV_SHUFFLE      = True            # Shuffle before splitting folds

# ---------------------------------------------------------------------------
# RANDOM SEARCH -- Number of iterations per kernel
# Try to keep this close to the number of Grid Search combinations
# so the comparison is fair computationally.
# ---------------------------------------------------------------------------

RANDOM_SEARCH_N_ITER = {
    "linear"  : 20,
    "rbf"     : 25,
    "poly"    : 30,
    "sigmoid" : 25,
}

# ---------------------------------------------------------------------------
# GRID SEARCH SPACES
#
# Each kernel has its own grid.
# Only the parameters relevant to that kernel are included.
# Adding a new kernel = adding a new entry here.
# ---------------------------------------------------------------------------

GRID_SEARCH_SPACES = {

    "linear": {
        # Linear SVM only has C as a meaningful parameter
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    },

    "rbf": {
        # RBF depends on both C (margin) and gamma (kernel width)
        "C"    : [0.1, 1.0, 10.0, 100.0],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
    },

    "poly": {
        # Polynomial has degree, C, gamma, and coef0
        "C"     : [0.1, 1.0, 10.0],
        "degree": [2, 3, 4],
        "gamma" : ["scale", "auto"],
        "coef0" : [0.0, 1.0],
    },

    "sigmoid": {
        # Sigmoid behaves like rbf but also has coef0
        "C"    : [0.1, 1.0, 10.0, 100.0],
        "gamma": ["scale", "auto", 0.001, 0.01],
        "coef0": [0.0, 0.5, 1.0],
    },
}

# ---------------------------------------------------------------------------
# RANDOM SEARCH DISTRIBUTIONS
#
# Each kernel has its own parameter distributions.
# loguniform = equal probability on log scale (good for C and gamma)
# uniform    = equal probability on linear scale (good for coef0)
# Lists      = discrete choices, sampled uniformly
# ---------------------------------------------------------------------------

RANDOM_SEARCH_DISTRIBUTIONS = {

    "linear": {
        "C": loguniform(0.01, 100),     # Samples between 0.01 and 100 on log scale
    },

    "rbf": {
        "C"    : loguniform(0.01, 100),
        "gamma": ["scale", "auto", 0.0001, 0.001, 0.01, 0.1],
    },

    "poly": {
        "C"     : loguniform(0.01, 100),
        "degree": [2, 3, 4],
        "gamma" : ["scale", "auto"],
        "coef0" : uniform(0, 2),        # Samples between 0.0 and 2.0
    },

    "sigmoid": {
        "C"    : loguniform(0.01, 100),
        "gamma": ["scale", "auto", 0.0001, 0.001, 0.01, 0.1],
        "coef0": uniform(-1, 2),        # Samples between -1.0 and 1.0
    },
}

# ---------------------------------------------------------------------------
# OUTPUT PATHS
# Search results go inside results/search/{method}/{kernel}_{timestamp}/
# Defined here so all three scripts use the same structure.
# ---------------------------------------------------------------------------

SEARCH_RESULTS_SUBDIR = "search"    # Inside RESULTS_DIR from paths.py


# ---------------------------------------------------------------------------
# BAYESIAN SEARCH SPACES
#
# Each kernel has its own list of skopt dimension objects.
# Real        = Continuous values sampled on log or uniform scale
# Integer     = Whole number values
# Categorical = Discrete choices
#
# The name field on each dimension must match the SVC parameter name exactly.
# ---------------------------------------------------------------------------

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
        Integer(2, 4, name="degree"),
        Categorical(["scale", "auto"], name="gamma"),
        Real(0.0, 2.0, prior="uniform", name="coef0"),
    ],

    "sigmoid": [
        Real(0.01, 100, prior="log-uniform", name="C"),
        Categorical(["scale", "auto", 0.001, 0.01, 0.1], name="gamma"),
        Real(-1.0, 1.0, prior="uniform", name="coef0"),
    ],
}

# ---------------------------------------------------------------------------
# BAYESIAN SEARCH -- Number of iterations per kernel
#
# Kept the same as RANDOM_SEARCH_N_ITER so the computational budget
# is equal across all three methods for a fair comparison.
#
# n_initial_points of those calls are random exploration before
# the Gaussian Process starts making informed suggestions.
# ---------------------------------------------------------------------------

BAYESIAN_N_CALLS = {
    "linear"  : 20,
    "rbf"     : 25,
    "poly"    : 30,
    "sigmoid" : 25,
}

# Number of random exploration steps before Gaussian Process takes over.
# Rule of thumb: roughly 20-25% of total n_calls.
BAYESIAN_N_INITIAL_POINTS = 5
