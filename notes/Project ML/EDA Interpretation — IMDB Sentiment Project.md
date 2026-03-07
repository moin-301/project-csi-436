
This note explains the **meaning of outputs produced during exploratory data analysis (EDA)** for the IMDB sentiment classification project.

It interprets each output and provides a **verdict about what it tells us about the dataset and preprocessing pipeline**.

Related notes:  
[[TF-IDF Vectorization]]  
[[Text Preprocessing Pipeline]]  
[[SVD]]  
[[t-SNE]]

---

# 1. Loading Libraries

## Output Meaning

The notebook imports the libraries required for analysis:

- `numpy` → numerical operations
    
- `matplotlib / seaborn` → visualization
    
- `scipy.sparse.load_npz` → loading sparse TF-IDF matrices
    
- `TruncatedSVD` → dimensionality reduction
    
- `TSNE` → nonlinear 2D visualization
    
- `pickle` → load saved TF-IDF vectorizer
    
- `Path` → file path handling
    

No scientific result is produced in this step.

## Verdict

Necessary setup for **data analysis and visualization**.

---

# 2. Processed Dataset Overview

Example output:

```
Train matrix : (25000, 20000)
Test matrix  : (25000, 20000)
Train labels : (25000,) | unique: [0 1]
Vocabulary   : 20,000 features
```

## What It Means

### TF-IDF Feature Matrix

```
(25000, 20000)
```

Meaning:

- **25,000 reviews**
    
- **20,000 vocabulary features**
    

Each review is represented as a **20,000-dimensional vector**.

Example representation:

```
review_i → [0,0,0.13,0,0,0.42,0,...]
```

Most values are zero because each review only contains a small number of words.

### Labels

```
unique: [0 1]
```

Meaning:

```
0 → Negative review
1 → Positive review
```

### Vocabulary Size

```
20,000 features
```

The TF-IDF vectorizer kept the **20k most informative words/bigrams**.

## Verdict

The raw text has been successfully converted into a **numerical feature space suitable for machine learning models like SVM**.

---

# 3. Dataset Balance

Example output:

```
Train   Positive: 12500   Negative: 12500
Test    Positive: 12500   Negative: 12500
```

## What It Means

The dataset is **perfectly balanced**.

```
Positive reviews = Negative reviews
```

Baseline accuracy:

```
50%
```

A classifier cannot cheat by predicting one class only.

## Verdict

This is an **ideal dataset for binary classification experiments**.

---

# 4. Raw vs Cleaned Review Example

Example:

### Raw Review

```
This movie was great! <br /><br /> Loved the acting.
```

### Cleaned Review

```
movie great loved acting
```

## What This Demonstrates

The preprocessing pipeline works correctly:

1. HTML removal
    
2. Lowercasing
    
3. Punctuation removal
    
4. Tokenization
    
5. Stopword removal
    

## Verdict

Confirms that **text preprocessing is functioning correctly**.

---

# 5. Review Length Distribution

Example statistics:

```
Min length  : 4 words
Max length  : 1420 words
Mean length : ~120 words
Median      : ~89 words
```

## Interpretation

Reviews vary widely in size.

Typical review:

```
~100 words
```

Distribution is **right-skewed**:

- Many short reviews
    
- Few extremely long reviews
    

Example:

Short reviews:

```
Great movie
Worst film ever
```

Long reviews:

```
Detailed story analysis
```

## Verdict

Dataset has a **natural language distribution typical of real user reviews**.

---

# 6. Most Important Words (Global TF-IDF)

Example output:

```
film
movie
story
character
people
```

## What This Means

These words have **high TF-IDF contribution across the dataset**.

However many of them are **neutral words**, appearing in both sentiments.

Example:

```
movie
film
story
```

They help describe the topic but not necessarily the sentiment.

## Verdict

Shows the **global vocabulary structure of the dataset**.

---

# 7. Word Clouds by Sentiment

Visualization:

Left → Positive Reviews  
Right → Negative Reviews

## Meaning

Word size represents **frequency within that sentiment class**.

Examples:

Positive words:

```
great
love
good
well
```

Negative words:

```
bad
worst
boring
```

## Interpretation

Positive and negative reviews tend to emphasize **different vocabularies**.

## Verdict

Confirms **clear linguistic patterns for sentiment**.

---

# 8. TF-IDF Matrix Sparsity

Example statistics:

```
Documents : 25,000
Features  : 20,000
Sparsity  : ~99%
```

## Explanation

Full matrix size:

```
25,000 × 20,000 = 500 million entries
```

But each review only contains a small subset of words.

Example:

```
review length ≈ 100 words
```

So only ~100 entries are non-zero.

Therefore the matrix is **extremely sparse**.

## Why Sparse Matrices Matter

Sparse storage:

```
stores only non-zero values
```

This drastically reduces memory usage.

## Verdict

Sparse representation is **necessary and correct for text datasets**.

---

# 9. Top TF-IDF Features for a Review

Example output:

```
branagh           0.316
shakespeare       0.302
talented cast     0.271
steals film       0.248
creditable source 0.231
```

## Interpretation

TF-IDF highlights **the most distinctive terms in that review**.

These words:

- appear frequently in this review
    
- appear rarely across other reviews
    

Therefore they characterize the review strongly.

## Verdict

Demonstrates how **TF-IDF captures document-specific importance**.

---

# 10. 2D SVD Projection

Output:

```
Total variance explained by 2 components: 0.45%
```

## Meaning

Original feature space:

```
20,000 dimensions
```

Reduced to:

```
2 dimensions
```

Most information is lost.

Therefore the visualization cannot perfectly separate classes.

## Verdict

Useful for **intuition**, but not reliable for classification conclusions.

---

# 11. t-SNE Visualization

Pipeline:

```
TF-IDF
↓
SVD (100 components)
↓
t-SNE
↓
2D visualization
```

Each point represents a review.

Colors:

```
Blue  → Positive review
Red   → Negative review
```

Observation:

- Lower region contains slightly more **positive reviews**
    
- Upper region contains slightly more **negative reviews**
    

However there is significant overlap.

## Why Overlap Happens

Language can contain mixed sentiment.

Example:

```
"The acting was great but the story was terrible."
```

So some reviews lie between classes.

## Verdict

The visualization shows **weak sentiment structure in feature space**, but not perfect separation.

---

# 12. Improved t-SNE Using More SVD Components

Example output:

```
Total variance explained by 100 SVD components: 7.41%
```

Increasing the number of SVD components preserves more information before t-SNE.

This leads to **clearer structure in the visualization**.

## Verdict

Better representation of **semantic relationships in high-dimensional text space**.

---

# 13. Preprocessing Pipeline Summary

Pipeline:

```
Raw IMDB Reviews
↓
HTML Removal
↓
Lowercasing
↓
Punctuation Removal
↓
Tokenization
↓
Stopword Removal
↓
TF-IDF Vectorization
↓
Feature Matrix (25000 × 20000)
```

## Meaning

The pipeline converts **raw text reviews into numerical feature vectors**.

These vectors are then used by machine learning models such as:

```
Support Vector Machines
```

## Verdict

A correct and standard **NLP preprocessing workflow**.

---

# Final Conclusion

The EDA confirms:

- Dataset integrity
    
- Balanced sentiment labels
    
- Correct preprocessing pipeline
    
- Sparse high-dimensional TF-IDF representation
    
- Linguistic differences between positive and negative reviews
    
- Partial sentiment structure in the feature space
    

This validates the dataset and preprocessing steps before **training machine learning models**.
