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

### Precision and recall
Calculated per class. For one class (say `sci.space`), every prediction falls into:

- **True positive (TP):** predicted `sci.space`, and it was `sci.space`
- **False positive (FP):** predicted `sci.space`, but it was another class
- **False negative (FN):** it was `sci.space`, but predicted as another class

From those:

- **Precision** = TP / (TP + FP). When the model says `sci.space`, how often is it right?
- **Recall** = TP / (TP + FN). Of all real `sci.space` posts, how many did it find?

They trade off. Predicting a class only when certain gives high precision and low recall.
Predicting it for everything gives perfect recall and poor precision.

### f1_score
Combines precision and recall into one per-class score:

```
F1 = 2 x (precision x recall) / (precision + recall)
```

This is the harmonic mean, which is pulled towards the weaker of the two. Precision 0.9 with
recall 0.1 has a plain average of 0.5 but an F1 of only 0.18, so a class scores well only if
it is good at both.

With several classes there is one F1 per class. The `average` parameter combines them:

- `average="macro"`: unweighted mean. Every class counts equally regardless of size.
- `average="weighted"`: mean weighted by each class's sample count. Large classes dominate.

Worked example with two classes:

| Class | Examples | F1 |
|---|---|---|
| `sci.space` | 900 | 0.95 |
| `talk.politics.misc` | 100 | 0.40 |

- `f1_macro` = (0.95 + 0.40) / 2 = **0.675**
- `f1_weighted` = (0.95 x 900 + 0.40 x 100) / 1000 = **0.895**

The weighted score looks healthy; the macro score exposes the weak class.

### Which metric to use
- **Accuracy** hides weak classes on imbalanced data. A fraud model that never flags fraud
  (around 0.1% of transactions) still scores about 99.9% accuracy.
- **`f1_macro`** is this platform's promotion metric (`primary_metric: f1_macro` in the model
  YAML). A candidate must improve average per-class quality to be promoted, not just get better
  at the common, easy classes.
- On balanced data (the 20 Newsgroups subset is roughly balanced) macro and weighted come out
  close. The gap opens on imbalanced data, which is exactly when it matters.

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
