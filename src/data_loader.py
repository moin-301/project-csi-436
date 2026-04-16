# =============================================================================
# data_loader.py
#
# PURPOSE : Load preprocessed data from disk.
#           This module does ONE job only -- reading data.
#           It does NOT clean, train, or evaluate anything.
#
# FILES IT READS (paths come from paths.py):
#   data/processed/train_tfidf.npz   -- sparse TF-IDF matrix
#   data/processed/train_labels.npy  -- 0/1 label array
#   data/processed/test_tfidf.npz
#   data/processed/test_labels.npy
# =============================================================================

import logging
import numpy as np
from scipy.sparse import load_npz

from paths import (
    TRAIN_MATRIX,
    TRAIN_LABELS,
    TEST_MATRIX,
    TEST_LABELS,
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("data_loader")


# =============================================================================
# LOAD FUNCTION
# =============================================================================

def load_data(split: str):
    """
    Load the TF-IDF matrix and labels for a given split.

    Parameters
    ----------
    split : str -- either "train" or "test"

    Returns
    -------
    X : scipy sparse matrix  -- shape (num_samples, num_features)
    y : numpy array          -- shape (num_samples,)  values are 0 or 1
                                0 = negative review
                                1 = positive review

    Example
    -------
    X_train, y_train = load_data("train")
    X_test,  y_test  = load_data("test")
    """

    # Pick the right file paths based on the split name
    if split == "train":
        matrix_path = TRAIN_MATRIX
        labels_path = TRAIN_LABELS
    elif split == "test":
        matrix_path = TEST_MATRIX
        labels_path = TEST_LABELS
    else:
        raise ValueError(
            f"Unknown split: '{split}'  --  choose 'train' or 'test'"
        )

    # Make sure both files exist before trying to read them
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Matrix file not found: {matrix_path}\n"
            f"Please run the preprocessing script first."
        )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}\n"
            f"Please run the preprocessing script first."
        )

    logger.info(f"Loading {split} data from disk...")

    X = load_npz(str(matrix_path))
    y = np.load(str(labels_path))

    num_positive = int(y.sum())
    num_negative = int(len(y) - num_positive)

    logger.info(f"  Split        : {split}")
    logger.info(f"  Matrix shape : {X.shape}  (samples x features)")
    logger.info(f"  Labels shape : {y.shape}")
    logger.info(f"  Positive (1) : {num_positive}")
    logger.info(f"  Negative (0) : {num_negative}")

    print(f"  [{split.upper()}]  Shape: {X.shape}  |  Positive: {num_positive}  |  Negative: {num_negative}")

    return X, y


# =============================================================================
# LOAD BOTH SPLITS AT ONCE
# =============================================================================

def load_all():
    """
    Convenience function to load both train and test in one call.

    Returns
    -------
    X_train, y_train, X_test, y_test

    Example
    -------
    X_train, y_train, X_test, y_test = load_all()
    """

    print("\nLoading train data...")
    X_train, y_train = load_data("train")

    print("Loading test data...")
    X_test, y_test = load_data("test")

    print("Data loading complete.\n")

    return X_train, y_train, X_test, y_test
