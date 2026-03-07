
## Definition

TF-IDF (Term Frequency – Inverse Document Frequency) is a method used to convert text into numerical features by measuring how important a word is in a document relative to a dataset.

It is widely used in [[Text Classification]] tasks such as [[Sentiment Analysis]].

---

# Motivation

Raw text cannot be used directly by machine learning models like [[Support Vector Machine]].

Example review:

```
This movie was fantastic
```

Models require numerical vectors:

```
[0.00, 0.32, 0.14, ...]
```

TF-IDF converts documents into such vectors.

---

# Core Idea

A word is important if:

```
It appears frequently in a document
BUT
It does not appear frequently across all documents
```

---

# Term Frequency (TF)

Measures how often a term appears in a document.

Basic formula:

$$
TF(t,d) = count(t,d)
$$

Often log-scaled:

$$
TF(t,d) = 1 + \log(count(t,d))
$$

This reduces the influence of very frequent words.

---

# Inverse Document Frequency (IDF)

Measures how rare a word is across the dataset.

$$
IDF(t) = \log\left(\frac{N}{df(t)}\right)
$$

Where:

```
N = total number of documents
df(t) = number of documents containing term t
```

Rare words receive higher weight.

---

# TF-IDF Score

Final weight:

$$
TF\text{-}IDF(t,d) = TF(t,d) \times IDF(t)
$$

A word receives a high TF-IDF score when:

```
frequent in the document
but rare across the dataset
```

---

# Example

Documents:

```
Doc1: movie fantastic fantastic
Doc2: movie boring
Doc3: movie good
```

Vocabulary:

```
movie
fantastic
boring
good
```

Since **movie** appears in every document, its TF-IDF weight is small.

Since **fantastic** appears rarely, its TF-IDF weight is high.

---

# Properties

TF-IDF produces a **sparse feature matrix**.

Example shape:

```
25000 documents × 20000 features
```

Most entries are zero.

Sparse matrices are stored efficiently using [[Sparse Matrix]] formats.

---

# Why TF-IDF Works Well for NLP

Text data is:

```
high dimensional
sparse
```

TF-IDF emphasizes meaningful words while down-weighting common words.

---

# Used In

- [[Sentiment Analysis]]
- [[Text Classification]]
- [[Information Retrieval]]
- [[Search Engines]]

---

# Used In This Project

See:

[[IMDB Sentiment Analysis — Preprocessing Pipeline]]