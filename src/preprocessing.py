import os
import re
import time
import logging
import pickle
from pathlib import Path

import nltk
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

# NLTK Downloads
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/preprocessing.log"),
    ],
)
logger = logging.getLogger("preprocessing")

# Constants
RAW_DATA_DIR    = Path("data/raw/aclImdb")
PROCESSED_DIR   = Path("data/processed")
VECTORIZER_PATH = PROCESSED_DIR / "vectorizer.pkl"
MAX_FEATURES    = 20_000
NGRAM_RANGE     = (1, 2)
MIN_DF          = 3
SUBLINEAR_TF    = True

STOP_WORDS = set(stopwords.words("english"))


# Step Functions

def remove_html(text: str) -> str:
    logger.debug("Removing HTML tags")
    return BeautifulSoup(text, "html.parser").get_text()


def lowercase(text: str) -> str:
    logger.debug("Lowercasing text")
    return text.lower()


def remove_punctuation(text: str) -> str:
    logger.debug("Removing punctuation")
    return re.sub(r"[^a-z\s]", "", text)


def tokenize(text: str) -> list:
    logger.debug("Tokenizing text")
    return word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    logger.debug("Removing stopwords")
    return [t for t in tokens if t not in STOP_WORDS]


def fit_vectorizer(corpus: list):
    logger.info(f"Fitting TF-IDF on training data | max_features={MAX_FEATURES}, ngram_range={NGRAM_RANGE}, min_df={MIN_DF}")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        sublinear_tf=SUBLINEAR_TF,
    )
    matrix = vectorizer.fit_transform(corpus)
    logger.info(f"TF-IDF matrix shape: {matrix.shape}")
    return matrix, vectorizer


def transform_vectorizer(corpus: list, vectorizer: TfidfVectorizer):
    logger.info("Transforming test data using fitted train vectorizer")
    matrix = vectorizer.transform(corpus)
    logger.info(f"TF-IDF matrix shape: {matrix.shape}")
    return matrix


# Single Review Pipeline

def clean_review(text: str) -> str:
    text   = remove_html(text)
    text   = lowercase(text)
    text   = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    return " ".join(tokens)


# Load Raw IMDB Data

def load_imdb(split: str) -> tuple:
    """
    split: 'train' or 'test'
    Returns (texts, labels) where label 1=positive, 0=negative
    """
    logger.info(f"Loading raw IMDB data - split: {split}")
    texts, labels = [], []
    for label, sentiment in enumerate(["neg", "pos"]):
        folder = RAW_DATA_DIR / split / sentiment
        files  = list(folder.glob("*.txt"))
        logger.info(f"  Found {len(files)} files in {folder}")
        for filepath in files:
            texts.append(filepath.read_text(encoding="utf-8"))
            labels.append(label)
    logger.info(f"Total loaded: {len(texts)} reviews")
    return texts, labels


# Master Pipeline

def preprocess_pipeline(split: str = "train"):
    start = time.time()
    logger.info("=" * 50)
    logger.info(f"Starting preprocessing pipeline - split: {split}")

    # 1. Load
    texts, labels = load_imdb(split)

    # 2. Clean each review
    logger.info("Cleaning reviews...")
    cleaned = [clean_review(t) for t in texts]

    empty = sum(1 for t in cleaned if t.strip() == "")
    if empty > 0:
        logger.warning(f"{empty} reviews were empty after cleaning")

    # 3. Vectorize
    # Train: fit and transform, then save the vectorizer
    # Test:  load the saved vectorizer and only transform, never refit
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if split == "train":
        matrix, vectorizer = fit_vectorizer(cleaned)
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(vectorizer, f)
        logger.info(f"Vectorizer saved to {VECTORIZER_PATH}")
    else:
        if not VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                "Vectorizer not found. Run preprocess_pipeline('train') first."
            )
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        logger.info(f"Vectorizer loaded from {VECTORIZER_PATH}")
        matrix = transform_vectorizer(cleaned, vectorizer)

    # 4. Save matrix, labels, and cleaned texts
    labels_array = np.array(labels)
    matrix_path  = PROCESSED_DIR / f"{split}_tfidf.npz"
    labels_path  = PROCESSED_DIR / f"{split}_labels.npy"
    cleaned_path = PROCESSED_DIR / f"{split}_cleaned.pkl"

    save_npz(str(matrix_path), matrix)
    np.save(str(labels_path), labels_array)
    with open(cleaned_path, "wb") as f:
        pickle.dump(cleaned, f)

    elapsed = time.time() - start
    logger.info(f"Saved matrix  : {matrix_path}")
    logger.info(f"Saved labels  : {labels_path}")
    logger.info(f"Saved cleaned : {cleaned_path}")
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info("=" * 50)

    return matrix, labels_array, vectorizer


# Main

if __name__ == "__main__":
    preprocess_pipeline(split="train")
    preprocess_pipeline(split="test")