## Goal
Convert raw IMDB movie reviews into **TF-IDF feature vectors** suitable for training machine learning models such as [[Support Vector Machine]].

---

# Pipeline Overview

Raw Reviews  
↓  
[[HTML Removal]]  
↓  
[[Lowercasing]]  
↓  
[[Punctuation Removal]]  
↓  
[[Tokenization]]  
↓  
[[Stopword Removal]]  
↓  
[[TF-IDF Vectorization]]  
↓  
Feature Matrix (25000 × 20000)

---

# 1. Imports and Environment Setup

## System Libraries

### `os`
Used for operating system interactions.

Examples:
- reading directories
- managing files
- interacting with environment variables

Example:

```python
os.listdir("data")
```

---

### `re`
Used for [[Regular Expressions]] to clean text.

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
Used to measure execution time of the preprocessing pipeline.

Example:

```python
start = time.time()
elapsed = time.time() - start
```

Useful for monitoring pipeline performance.

---

### `logging`
Provides **structured logging** instead of print statements.

Benefits:

- timestamps
- log levels
- persistent logs for debugging

---

### `pickle`
Used to **serialize Python objects**.

In this project it saves the trained [[TF-IDF Vectorizer]].

```python
pickle.dump(vectorizer, f)
```

This preserves:

- vocabulary
- IDF weights
- vectorizer configuration

---

### `Path` (from `pathlib`)
Used for modern file path handling.

Example:

```python
RAW_DATA_DIR / "train" / "pos"
```

Advantages:

- cleaner syntax
- cross-platform compatibility
- safer file manipulation

---

# 2. NLP and Machine Learning Libraries

### `nltk`
The **Natural Language Toolkit** used for basic NLP operations.

Used for:

- [[Tokenization]]
- [[Stopword Removal]]

---

### `numpy`
Provides efficient numerical arrays.

Used to store labels:

```python
labels_array = np.array(labels)
```

---

### `BeautifulSoup`
Used for [[HTML Removal]] in IMDB reviews.

Example:

```python
BeautifulSoup(text, "html.parser").get_text()
```

Converts:

```
Great movie<br /><br />Loved it
```

into

```
Great movie Loved it
```

---

### `stopwords`
Provides a list of common English stopwords.

Examples:

```
the
is
and
to
a
```

These words usually carry **little discriminative information** for classification.

---

### `word_tokenize`
Splits text into individual words.

Example:

```
"This movie was great!"
```

becomes

```
["This", "movie", "was", "great"]
```

---

### `save_npz`
Used to store **sparse matrices**.

TF-IDF matrices are sparse because most entries are zero.

```python
save_npz("train_tfidf.npz", matrix)
```

---

### `TfidfVectorizer`
Converts text into numerical features using [[TF-IDF]].

Formula:

$$
TF\text{-}IDF = TF \times IDF
$$

Where

- TF = term frequency
- IDF = inverse document frequency

Example vector:

```
movie → 0.12
great → 0.31
boring → 0.44
```

Each review becomes a **feature vector**.

---

# 3. NLTK Resource Downloads

```python
nltk.download("punkt")
nltk.download("stopwords")
```

Required resources:

### `punkt`
Required for [[Tokenization]] using `word_tokenize`.

---

### `stopwords`
Provides the stopword list used for filtering.

---

⚠ Note

```
nltk.download("punkt_tab")
```

is unnecessary and should be removed.

---

# 4. Logging System

Logging replaces `print()` with structured messages.

Configuration:

```python
logging.basicConfig(...)
```

---

## Log Level

```python
level=logging.INFO
```

Records messages of severity:

```
INFO
WARNING
ERROR
```

---

## Log Format

```
%(asctime)s | %(levelname)s | %(name)s | %(message)s
```

Example:

```
2026-03-06 22:03:29 | INFO | preprocessing | Cleaning reviews...
```

---

## Handlers

### StreamHandler
Writes logs to the **terminal**.

### FileHandler

```
logs/preprocessing.log
```

Stores logs for later debugging.

---

# 5. Output Artifacts

After preprocessing the following files are created.

```
data/processed/
```

### TF-IDF Matrices

```
train_tfidf.npz
test_tfidf.npz
```

Sparse matrices:

```
documents × features
```

Example:

```
(25000 × 20000)
```

---

### Labels

```
train_labels.npy
test_labels.npy
```

Label encoding:

```
0 → negative
1 → positive
```

---

### Vectorizer

```
vectorizer.pkl
```

Stores:

- vocabulary
- IDF weights
- preprocessing configuration

Ensures **test data uses the same vocabulary as training data**.

---

# Key Idea

This preprocessing pipeline converts

```
Raw movie reviews
```

into

```
TF-IDF feature vectors
```

which can be directly used by machine learning algorithms such as [[Support Vector Machine]].