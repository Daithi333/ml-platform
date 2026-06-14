# Scikit-Learn Concepts

Reference notes on the scikit-learn components used in this project, starting with the
text classification pipeline.

## Pipeline

A `Pipeline` chains preprocessing and model steps into a single object. You call `.fit()`
and `.predict()` on it as if it were one model, but internally it runs each step in
sequence.

```python
Pipeline([
    ("tfidf", TfidfVectorizer(...)),    # Step 1: transform text -> numbers
    ("classifier", LogisticRegression(...)),  # Step 2: classify
])
```

Why pipelines matter:
- The entire chain is serializable (save/load as one artifact in MLflow)
- No risk of data leakage between train/test (preprocessing is fit only on training data)
- Clean interface: raw text in, predictions out

## TfidfVectorizer

Converts raw text into numerical features that a classifier can consume.

**TF** (Term Frequency): How often a word appears in a document.
**IDF** (Inverse Document Frequency): Penalises words that appear in many documents
(e.g. "the", "is") since they carry less signal.

**TF-IDF score** = TF x IDF. Words that are frequent in one document but rare across the
corpus get high scores. Words that appear everywhere get low scores.

Key parameters:
- `max_features`: Vocabulary cap. Only keeps the N most frequent terms. Controls
  dimensionality (10,000 is a reasonable default for medium corpora).
- `ngram_range`: `(1, 2)` means use both individual words (unigrams) and pairs of
  consecutive words (bigrams). Bigrams capture phrases like "machine learning" that
  lose meaning when split.
- `stop_words`: `"english"` removes common words (the, is, at) before computing TF-IDF.

Output: a sparse matrix of shape `(n_documents, max_features)` where each cell is the
TF-IDF score for that term in that document.

## LogisticRegression

Despite the name, this is a classification algorithm (not regression). It models the
probability of each class given the input features.

For multi-class problems (5 newsgroup categories), it uses a softmax function to produce
a probability distribution across all classes. The predicted class is the one with the
highest probability.

Key parameters:
- `C`: Inverse regularisation strength. Smaller C = more regularisation (simpler model,
  less overfitting). Larger C = less regularisation (fits training data more closely).
  Default 1.0 is a good starting point.
- `max_iter`: Maximum iterations for the optimiser to converge. Increase if you get
  convergence warnings. 1000 is safe for most text classification tasks.
- `solver`: `"lbfgs"` is the default for multi-class. Fast, memory-efficient.
- `multi_class`: Removed in sklearn 1.7 (now always multinomial automatically).

Output: `.predict()` returns class indices, `.predict_proba()` returns probability
distributions.

## train_test_split

Splits data into training and test sets. The model trains on one subset and is evaluated
on the other (data it has never seen).

Key parameters:
- `test_size`: Fraction held out for evaluation (0.2 = 20% test, 80% train).
- `random_state`: Seed for reproducibility. Same seed = same split every time.
- `stratify`: Ensures the class distribution in the split matches the original. Critical
  for imbalanced datasets (e.g. if 90% of data is one category).

## Evaluation Metrics

### accuracy_score
Proportion of correct predictions. Simple but misleading on imbalanced data (predicting
the majority class always gives high accuracy).

### f1_score
Harmonic mean of precision and recall. Better than accuracy for imbalanced classes.

- `average="macro"`: Compute F1 per class, then take the unweighted mean. Treats all
  classes equally regardless of size.
- `average="weighted"`: Compute F1 per class, weighted by number of samples in each
  class. Accounts for class imbalance.

### classification_report
Prints precision, recall, F1, and support (sample count) for each class. The go-to
output for understanding per-class performance.

## 20 Newsgroups Dataset

A classic text classification benchmark bundled with scikit-learn. ~18,000 posts across
20 newsgroup topics. We subset to 5 categories to keep training fast (the model isn't
the point -- the platform is).

`remove=("headers", "footers", "quotes")` strips metadata that would let the model cheat
(e.g. a newsgroup header literally containing the category name).

## How It All Connects

```
Raw text ["NASA launched..."]
    |
    v
TfidfVectorizer.transform()  -->  sparse matrix [0.0, 0.3, 0.0, 0.7, ...]
    |
    v
LogisticRegression.predict_proba()  -->  [0.02, 0.01, 0.05, 0.91, 0.01]
    |
    v
argmax  -->  index 3  -->  "sci.space"
```

The pipeline encapsulates this entire flow. MLflow serializes the whole pipeline as a
single artifact. The serving layer loads it and calls `.predict()` on raw text strings --
no preprocessing code needed at inference time.

## What Changes When You Swap Datasets

Nothing in the pipeline code changes. The YAML config specifies:
- Which dataset to load (source + name)
- Which categories to use (labels)
- Hyperparameters (max_features, ngram_range, C)

Same `text_classifier.py` architecture, different config file, different registered model
in MLflow. That's the platform design.
