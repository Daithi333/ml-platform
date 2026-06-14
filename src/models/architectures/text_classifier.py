"""Text classifier architecture.

A generic TF-IDF + classifier pipeline. Can be used for any text classification
problem — newsgroups, support tickets, content moderation, etc.

The specific algorithm and parameters come from the model config YAML.
"""

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_pipeline(params: dict[str, Any]) -> Pipeline:
    """Build a text classification pipeline from config parameters.

    Supported params:
        max_features: int (default 10000)
        ngram_range: [min, max] (default [1, 2])
        max_iter: int (default 1000)
        C: float (default 1.0)
        algorithm: str (default 'logistic_regression')
    """
    max_features = params.get("max_features", 10000)
    ngram_range = tuple(params.get("ngram_range", [1, 2]))
    max_iter = params.get("max_iter", 1000)
    c_value = params.get("C", 1.0)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
    )

    classifier = LogisticRegression(
        max_iter=max_iter,
        C=c_value,
        solver="lbfgs",
    )

    return Pipeline(
        [
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ]
    )
