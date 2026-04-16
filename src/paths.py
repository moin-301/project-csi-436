# =============================================================================
# paths.py
#
# PURPOSE : One single place for ALL directory and file paths.
#           If you give this project to someone else, they only need to
#           edit THIS file to make everything work on their machine.
#
# RULE    : No paths are hardcoded anywhere else in the project.
#           Every other file imports from here.
# =============================================================================

from pathlib import Path

# ---------------------------------------------------------------------------
# ROOT
# Change BASE_DIR if your project lives somewhere else.
# Everything else is relative to it, so nothing else needs to change.
# ---------------------------------------------------------------------------

BASE_DIR = Path(".")

# ---------------------------------------------------------------------------
# INPUT DATA
# Where the preprocessed .npz and .npy files live.
# ---------------------------------------------------------------------------

DATA_DIR        = BASE_DIR / "data" / "processed"

TRAIN_MATRIX    = DATA_DIR / "train_tfidf.npz"
TRAIN_LABELS    = DATA_DIR / "train_labels.npy"

TEST_MATRIX     = DATA_DIR / "test_tfidf.npz"
TEST_LABELS     = DATA_DIR / "test_labels.npy"

# ---------------------------------------------------------------------------
# OUTPUT
# All results go inside results/.
# Each training run gets its own timestamped subfolder inside results/.
# ---------------------------------------------------------------------------

RESULTS_DIR     = BASE_DIR / "results"

# ---------------------------------------------------------------------------
# LOGS
# One log file for the whole pipeline session.
# ---------------------------------------------------------------------------

LOGS_DIR        = BASE_DIR / "logs"
LOG_FILE        = LOGS_DIR / "pipeline.log"
