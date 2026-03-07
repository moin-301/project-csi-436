# 1. Purpose of Imports

## System and Utility Libraries

### `os`

Used for **operating system interactions** such as:

- reading directories
    
- managing file paths
    
- interacting with environment variables
    

Example use:

```python
os.listdir("data")
```

---

### `re`

Used for **regular expressions** to clean text.

Example:

```python
re.sub(r"[^a-z\s]", "", text)
```

Removes:

- punctuation
    
- numbers
    
- special characters
    

---

### `time`

Used for **measuring execution time** of the pipeline.

Example:

```python
start = time.time()
elapsed = time.time() - start
```

Useful for monitoring preprocessing performance.

---

### `logging`

Used for **structured logging instead of print statements**.

Provides:

- timestamps
    
- log levels
    
- persistent logs for debugging
    

---

### `pickle`

Used for **saving and loading Python objects**.

In this project it saves the trained TF-IDF vectorizer:

```python
pickle.dump(vectorizer, f)
```

This preserves:

- vocabulary
    
- IDF weights
    
- vectorizer configuration
    

---

### `Path` (from `pathlib`)

Provides **modern file path handling**.

Example:

```python
RAW_DATA_DIR / "train" / "pos"
```

Advantages:

- cleaner syntax
    
- cross-platform compatibility
    
- easier file operations
    

---

# 2. NLP and Machine Learning Libraries

### `nltk`

The **Natural Language Toolkit** for text processing.

Used for:

- tokenization
    
- stopword lists
    

---

### `numpy`

Provides **efficient numerical arrays**.

Used to store labels:

```python
labels_array = np.array(labels)
```

---

### `BeautifulSoup`

Used to **remove HTML tags** from IMDB reviews.

Example:

```python
BeautifulSoup(text, "html.parser").get_text()
```

Converts:

```
Great movie<br /><br />Loved it
```

into:

```
Great movie Loved it
```

---

### `stopwords` (from NLTK)

Provides a list of **common English stopwords**.

Examples:

```
the
is
and
to
a
```

These are removed because they usually carry little semantic meaning.

---

### `word_tokenize`

Splits text into individual words (tokens).

Example:

```
"This movie was great!"
```

becomes:

```
["This", "movie", "was", "great"]
```

---

### `save_npz` (from `scipy.sparse`)

Used to **save sparse matrices**.

TF-IDF matrices are sparse because most entries are zero.

Example:

```python
save_npz("train_tfidf.npz", matrix)
```

This saves memory compared to dense matrices.

---

### `TfidfVectorizer`

Converts text into **numerical features using TF-IDF**.

Formula:

$$  
TF\text{-}IDF = TF \times IDF  
$$

Where:

- TF = term frequency in a document
    
- IDF = inverse document frequency across documents
    

Example vector:

```
movie → 0.12
great → 0.31
boring → 0.44
```

Each review becomes a **numerical feature vector**.

---

# 3. NLTK Resource Downloads

```python
nltk.download("punkt")
nltk.download("stopwords")
```

These commands download required NLP resources.

### `punkt`

Required for **tokenization** using:

```python
word_tokenize()
```

---

### `stopwords`

Downloads the English stopword list used for filtering.

---

⚠ Note:

```
nltk.download("punkt_tab")
```

is **not required** and can be removed.

---

# 4. Logging Configuration

Logging replaces `print()` with structured log messages.

Configuration:

```python
logging.basicConfig(...)
```

---

## Log Level

```python
level=logging.INFO
```

Records messages with severity:

```
INFO
WARNING
ERROR
```

but ignores debug messages.

---

## Log Format

```python
format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
```

Example output:

```
2026-03-06 22:03:29 | INFO | preprocessing | Cleaning reviews...
```

Components:

|Field|Meaning|
|---|---|
|`asctime`|timestamp|
|`levelname`|severity level|
|`name`|logger name|
|`message`|log content|

---

## Date Format

```python
datefmt="%Y-%m-%d %H:%M:%S"
```

Example timestamp:

```
2026-03-06 22:03:29
```

---

## Handlers

Handlers define **where log messages are written**.

### StreamHandler

Outputs logs to the **terminal**.

---

### FileHandler

```python
logging.FileHandler("logs/preprocessing.log")
```

Writes logs to:

```
logs/preprocessing.log
```

Useful for experiment tracking and debugging.

---

## Logger Creation

```python
logger = logging.getLogger("preprocessing")
```

Later used as:

```python
logger.info(...)
logger.warning(...)
logger.debug(...)
```

Example output:

```
INFO | preprocessing | Loading raw IMDB data
```

---

# Key Takeaway

This code section prepares the **environment for the preprocessing pipeline** by:

- importing NLP and ML libraries
    
- downloading required NLTK resources
    
- configuring a structured logging system
    

Logging ensures that the **entire preprocessing pipeline can be monitored, debugged, and reproduced easily**.



# Text Preprocessing Pipeline — Constants and Cleaning Functions

## Overview

This section defines:

1. **Configuration constants**
    
2. **Text cleaning functions**
    
3. **TF-IDF feature generation functions**
    
4. **A pipeline to process a single review**
    

These components prepare raw movie reviews for **machine learning models such as SVM**.

---

# 4. Constants (Configuration Parameters)

These values define **how preprocessing and vectorization behave**.

## Dataset Paths

```python
RAW_DATA_DIR = Path("data/raw/aclImdb")
PROCESSED_DIR = Path("data/processed")
```

Purpose:

- `RAW_DATA_DIR` → location of the **original IMDB dataset**
    
- `PROCESSED_DIR` → location to store **processed features**
    

Example structure:

```
data/
   raw/
      aclImdb/
   processed/
      train_tfidf.npz
      test_tfidf.npz
```

---

## Vectorizer Storage

```python
VECTORIZER_PATH = PROCESSED_DIR / "vectorizer.pkl"
```

Purpose:

Store the trained **TF-IDF vectorizer**.

Why this matters:

The **same vocabulary must be used for training and test data** to avoid data leakage.

---

## TF-IDF Configuration

```python
MAX_FEATURES = 20_000
```

Limits vocabulary size.

Meaning:

Only the **20,000 most informative words/ngrams** are kept.

Benefit:

- reduces dimensionality
    
- improves training speed
    
- removes extremely rare terms
    

---

```python
NGRAM_RANGE = (1, 2)
```

Defines the type of text features.

Meaning:

- `(1,1)` → single words (unigrams)
    
- `(1,2)` → words + word pairs
    

Example:

```
"great movie"

unigram: great, movie
bigram: great movie
```

Bigrams help capture **local context**.

---

```python
MIN_DF = 3
```

Minimum document frequency.

Meaning:

A term must appear in **at least 3 reviews** to be included.

Purpose:

Removes extremely rare words.

Example removed terms:

```
misspellings
very uncommon names
noise tokens
```

---

```python
SUBLINEAR_TF = True
```

Applies **log scaling** to term frequency.

Instead of:

$$  
TF = count  
$$

Use:

$$  
TF = 1 + \log(count)  
$$

Benefit:

Reduces the influence of very frequent words.

---

## Stopword Set

```python
STOP_WORDS = set(stopwords.words("english"))
```

Creates a **set of common English stopwords**.

Examples:

```
the
is
and
to
of
```

Using a **set** makes lookup faster:

```
O(1) membership check
```

instead of list scanning.

---

# 5. Text Cleaning Functions

These functions clean and normalize the raw text.

Each step performs **one transformation**, which improves modularity.

---

## Remove HTML

```python
def remove_html(text: str) -> str:
```

Purpose:

Remove HTML tags present in IMDB reviews.

Example:

```
"This movie was great <br /><br /> Loved it"
```

becomes

```
"This movie was great  Loved it"
```

Uses:

```
BeautifulSoup HTML parser
```

---

## Lowercasing

```python
def lowercase(text: str) -> str:
```

Converts all text to lowercase.

Example:

```
Great Movie → great movie
```

Purpose:

Prevents duplicate tokens.

Without lowercasing:

```
Movie
movie
MOVIE
```

would be treated as different features.

---

## Remove Punctuation

```python
def remove_punctuation(text: str) -> str:
```

Uses a **regular expression**:

```
[^a-z\s]
```

Meaning:

Remove anything that is not:

- lowercase letters
    
- whitespace
    

Example:

```
"This movie!!! 10/10"
```

becomes

```
this movie
```

---

## Tokenization

```python
def tokenize(text: str) -> list:
```

Splits text into individual words.

Example:

```
"this movie is great"
```

becomes

```
["this", "movie", "is", "great"]
```

Tokenization is required for **word-level processing**.

---

## Stopword Removal

```python
def remove_stopwords(tokens: list) -> list:
```

Removes common words that carry little meaning.

Example:

```
["this", "movie", "is", "great"]
```

becomes

```
["movie", "great"]
```

Purpose:

Reduce noise and dimensionality.

---

# 6. TF-IDF Feature Construction

Two functions manage vectorization.

---

## Fit Vectorizer (Training Data)

```python
def fit_vectorizer(corpus: list):
```

This step **learns the vocabulary** and computes IDF values.

Key operation:

```python
matrix = vectorizer.fit_transform(corpus)
```

Meaning:

- build vocabulary
    
- compute IDF weights
    
- convert documents into vectors
    

Output:

```
matrix shape = (documents × features)
```

Example:

```
(25000, 20000)
```

Returns:

```
TF-IDF matrix
trained vectorizer
```

---

## Transform Vectorizer (Test Data)

```python
def transform_vectorizer(corpus, vectorizer)
```

Uses the **existing vocabulary** learned during training.

Operation:

```
vectorizer.transform()
```

Important principle:

Test data must **not modify the vocabulary**.

---

# 7. Single Review Cleaning Pipeline

```python
def clean_review(text: str) -> str:
```

Applies all cleaning steps sequentially.

Pipeline:

```
raw review
   ↓
remove HTML
   ↓
lowercase
   ↓
remove punctuation
   ↓
tokenize
   ↓
remove stopwords
   ↓
join tokens
```

Final output example:

```
movie absolutely terrible acting bad plot
```

This cleaned text is ready for **TF-IDF vectorization**.

---

# Key Concept

The preprocessing pipeline converts:

```
Raw text reviews
```

into

```
Clean normalized text
```

which can then be transformed into

```
numerical feature vectors (TF-IDF)
```

for machine learning models such as **Support Vector Machines**.


# 8. Loading IMDB Dataset — `load_imdb()`

## Purpose

This function **loads the raw IMDB movie review dataset** from disk and converts it into two Python lists:

- `texts` → the review contents
    
- `labels` → the sentiment labels
    

Output format:

```
texts  = [review1, review2, review3, ...]
labels = [0, 1, 0, ...]
```

Where:

```
0 → negative review
1 → positive review
```

---

# Function Definition

```python
def load_imdb(split: str) -> tuple:
```

## Parameters

`split`

```
"train"
or
"test"
```

This determines which portion of the dataset is loaded.

Example directory structure:

```
data/raw/aclImdb/

    train/
        pos/
        neg/

    test/
        pos/
        neg/
```


# Logging the Start of Loading

```python
logger.info(f"Loading raw IMDB data - split: {split}")
```

Produces log output like:

```
INFO | Loading raw IMDB data - split: train
```

Purpose:

- track pipeline progress
    
- debugging
    
- experiment monitoring
    

---

# Initialize Storage Lists

```python
texts, labels = [], []
```

Two lists are created:

|Variable|Purpose|
|---|---|
|`texts`|store review text|
|`labels`|store sentiment label|

Example later:

```
texts = [
    "This movie was amazing",
    "Worst movie ever"
]

labels = [
    1,
    0
]
```

---

# Iterating Over Sentiment Classes

```python
for label, sentiment in enumerate(["neg", "pos"]):
```

This loop processes **two folders**:

```
neg
pos
```

`enumerate()` assigns numeric labels automatically.

Result:

|sentiment|label|
|---|---|
|neg|0|
|pos|1|

---

# Construct Dataset Folder Path

```python
folder = RAW_DATA_DIR / split / sentiment
```

Example if:

```
split = "train"
sentiment = "neg"
```

Result:

```
data/raw/aclImdb/train/neg
```


---

# Listing Review Files

```python
files = list(folder.glob("*.txt"))
```

Purpose:

Find all review files in the folder.

Example result:

```
0_3.txt
1_4.txt
2_1.txt
...
```

Each file contains **one movie review**.

---

# Logging File Count

```python
logger.info(f"  Found {len(files)} files in {folder}")
```

Example output:

```
Found 12500 files in data/raw/aclImdb/train/neg
```

This confirms the dataset loaded correctly.

Expected IMDB counts:

```
train/neg → 12500
train/pos → 12500
test/neg  → 12500
test/pos  → 12500
```

---

# Reading Each Review

```python
for filepath in files:
```

Iterates over every `.txt` file.

---

# Loading Review Text

```python
texts.append(filepath.read_text(encoding="utf-8"))
```

Reads the contents of the file.

Example file:

```
This movie was terrible. I hated it.
```

This string is added to the `texts` list.

---

# Assigning Labels

```python
labels.append(label)
```

The numeric label (0 or 1) is stored.

Example:

```
texts  = ["Great movie", "Terrible film"]
labels = [1, 0]
```

The order of texts and labels **matches exactly**.

---

# Logging Total Reviews Loaded

```python
logger.info(f"Total loaded: {len(texts)} reviews")
```

Expected output:

```
Total loaded: 25000 reviews
```

Because IMDB training set contains:

```
12500 positive
12500 negative
```

---

# Returning the Dataset

```python
return texts, labels
```

Return type:

```
tuple
```

Contents:

```
(texts, labels)
```

Example structure:

```
texts = [
   "This movie was great...",
   "Worst movie ever...",
   ...
]

labels = [
   1,
   0,
   ...
]
```

These are then used for **text preprocessing and TF-IDF vectorization**.

---

# Data Flow After This Step

The output feeds into the preprocessing pipeline:

```
load_imdb()
        ↓
clean_review()
        ↓
TF-IDF vectorization
        ↓
feature matrix
```

---

# Key Idea

This function converts the dataset from **file-based format**

```
text files on disk
```

into a **machine learning friendly format**

```
list of texts + list of labels
```

which can then be processed by NLP and ML algorithms.

---

# 9. Preprocessing Controller — `preprocess_pipeline()`

## Purpose

This function is the **main controller of the preprocessing workflow**.  
It performs the complete pipeline:

```
Load dataset
   ↓
Clean reviews
   ↓
Convert text → TF-IDF features
   ↓
Save processed data
```

The function ensures that **training and test data are processed correctly without data leakage**.

---

# 10. Function Definition

```python
def preprocess_pipeline(split: str = "train"):
```

### Parameter

`split`

```
"train"  → training dataset
"test"   → test dataset
```

Default value:

```
"train"
```

---

# 10.1 Start Time Measurement

```python
start = time.time()
```

Purpose:

Measure the **total runtime of the pipeline**.

Later:

```
elapsed = time.time() - start
```

Example output:

```
Pipeline complete in 20.0s
```

---

# 10.2 Log Pipeline Start

```python
logger.info("=" * 50)
logger.info(f"Starting preprocessing pipeline - split: {split}")
```

Produces logs like:

```
==================================================
Starting preprocessing pipeline - split: train
```

Purpose:

- visually separate pipeline runs
    
- easier debugging
    

---

# 10.3 Load Dataset

```python
texts, labels = load_imdb(split)
```

This calls the previously defined function.

Output:

```
texts  = list of reviews
labels = list of sentiment labels
```

Example:

```
texts  = ["Great movie...", "Worst film..."]
labels = [1, 0]
```

---

# 10.4 Clean Each Review

```python
cleaned = [clean_review(t) for t in texts]
```

This applies the **text cleaning pipeline** to every review.

Cleaning steps include:

```
remove HTML
↓
lowercase
↓
remove punctuation
↓
tokenize
↓
remove stopwords
```

Example transformation:

```
Raw review:
"This movie was GREAT!!! <br /><br /> Loved it."

Cleaned review:
movie great loved
```

---

# 10.5 Detect Empty Reviews

```python
empty = sum(1 for t in cleaned if t.strip() == "")
```

Purpose:

Check if cleaning removed **all content** from some reviews.

Example problematic case:

```
"the and is"
```

After stopword removal:

```
""
```

If empty reviews exist:

```python
logger.warning(...)
```

This prevents **silent data loss**.

---

# 10.6 Ensure Processed Data Folder Exists

```python
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
```

Purpose:

Create the directory:

```
data/processed/
```

Arguments:

```
parents=True     → create parent directories if needed
exist_ok=True    → do not crash if folder already exists
```

---

# 10.7 TF-IDF Vectorization

The behavior depends on whether we process **training or test data**.

---

## Training Mode

```python
if split == "train":
```

Training must:

```
learn vocabulary
compute IDF values
```

### Fit TF-IDF Model

```python
matrix, vectorizer = fit_vectorizer(cleaned)
```

Result:

```
matrix shape = (25000, 20000)
```

Meaning:

```
documents = 25,000
features  = 20,000
```

---

### Save Vectorizer

```python
pickle.dump(vectorizer, f)
```

File saved:

```
data/processed/vectorizer.pkl
```

Why?

The same vocabulary must be used for **test data transformation**.

---

## Test Mode

```python
else:
```

Test data must **never modify the vocabulary**.

---

### Ensure Vectorizer Exists

```python
if not VECTORIZER_PATH.exists():
```

If missing:

```
FileNotFoundError
```

This ensures the **training pipeline runs first**.

---

### Load Saved Vectorizer

```python
vectorizer = pickle.load(f)
```

Now the exact training vocabulary is restored.

---

### Transform Test Data

```python
matrix = transform_vectorizer(cleaned, vectorizer)
```

Important rule:

```
fit()      → only training data
transform() → test data
```

This prevents **data leakage**.

---

# 10.8 Save Processed Outputs

Three files are saved.

---

## TF-IDF Matrix

```python
save_npz(matrix_path, matrix)
```

File:

```
train_tfidf.npz
test_tfidf.npz
```

Contains:

```
documents × features sparse matrix
```

---

## Labels

```python
np.save(labels_path, labels_array)
```

Files:

```
train_labels.npy
test_labels.npy
```

Contains:

```
0 → negative
1 → positive
```

---

## Cleaned Text

```python
pickle.dump(cleaned, f)
```

Files:

```
train_cleaned.pkl
test_cleaned.pkl
```

Purpose:

Allows later analysis such as:

```
EDA
word frequency
word clouds
visualizations
```

---

# 10.9 Log Saved Outputs

Example logs:

```
Saved matrix  : data/processed/train_tfidf.npz
Saved labels  : data/processed/train_labels.npy
Saved cleaned : data/processed/train_cleaned.pkl
```

This confirms that preprocessing finished successfully.

---

# 10.10 Log Pipeline Completion

```python
logger.info(f"Pipeline complete in {elapsed:.1f}s")
```

Example:

```
Pipeline complete in 20.0s
```

---

# 10.11 Return Values

```python
return matrix, labels_array, vectorizer
```

Returned objects:

|Variable|Meaning|
|---|---|
|matrix|TF-IDF feature matrix|
|labels_array|sentiment labels|
|vectorizer|trained TF-IDF model|

These are used for **training machine learning models**.

---

# Complete Data Flow

```
Raw IMDB reviews
        ↓
load_imdb()
        ↓
clean_review()
        ↓
TF-IDF vectorization
        ↓
Sparse feature matrix
        ↓
Saved files for ML training
```

---

# Key Idea

This function converts **raw text data**

```
movie reviews
```

into **numerical feature matrices**

```
TF-IDF vectors
```

which can be directly used by machine learning algorithms such as **Support Vector Machines (SVM)**.