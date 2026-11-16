"""Classical baselines: TF-IDF features + a linear classifier.

Each model is a scikit-learn ``Pipeline`` (vectoriser -> classifier). Putting
the vectoriser INSIDE the pipeline is a leakage guard: ``pipeline.fit(train)``
learns the vocabulary and IDF weights from training text only, and the saved
pipeline applies exactly the same transformation at inference time.

Class imbalance
---------------
* Logistic Regression / Linear SVM use ``class_weight="balanced"``: each class's
  errors are weighted by n_samples / (n_classes * n_class_samples), so a mistake
  on a hate-speech tweet costs roughly 13x more than on an offensive one.
* Multinomial Naive Bayes has no class_weight; its class priors are learned from
  the data, so expect it to under-predict the minority class. That weakness is
  part of what the comparison is meant to show.

LinearSVC outputs margins, not probabilities. It is wrapped in
``CalibratedClassifierCV`` (Platt scaling with internal cross-validation on the
training data) so the app and LIME can use ``predict_proba`` for every model.
"""
from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.constants import BASELINE_MODELS

PARAM_NAMES = {"logistic_regression": "C", "linear_svm": "C", "naive_bayes": "alpha"}


def build_vectorizer(tfidf_cfg: dict) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,                                   # text is already cleaned
        ngram_range=tuple(tfidf_cfg.get("ngram_range", (1, 2))),
        min_df=tfidf_cfg.get("min_df", 2),
        max_df=tfidf_cfg.get("max_df", 0.95),
        max_features=tfidf_cfg.get("max_features", 50000),
        sublinear_tf=tfidf_cfg.get("sublinear_tf", True),  # 1 + log(tf): damps repeated words
        token_pattern=r"(?u)\b\w+\b",                      # keep one-letter tokens such as "u"
    )


def build_pipeline(model_name: str, tfidf_cfg: dict, param: float, seed: int = 42) -> Pipeline:
    if model_name == "logistic_regression":
        classifier = LogisticRegression(C=param, class_weight="balanced", max_iter=2000, random_state=seed)
    elif model_name == "linear_svm":
        classifier = CalibratedClassifierCV(
            LinearSVC(C=param, class_weight="balanced", random_state=seed), cv=3
        )
    elif model_name == "naive_bayes":
        classifier = MultinomialNB(alpha=param)
    else:
        raise ValueError(f"Unknown baseline '{model_name}'. Choose from {BASELINE_MODELS}")
    return Pipeline([("tfidf", build_vectorizer(tfidf_cfg)), ("clf", classifier)])
