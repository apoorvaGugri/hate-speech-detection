"""Metrics and result persistence shared by every model.

Why macro-F1 is the headline metric
-----------------------------------
About three quarters of the dataset is "offensive language". A model that
always answers "offensive" would score ~77% accuracy while never detecting a
single hate-speech tweet. Macro-F1 averages the F1 of each class with equal
weight, so the small hate-speech class counts as much as the big one.
Weighted-F1 (weighted by class size) is reported too, but behaves like accuracy.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.constants import ID2LABEL, LABEL_IDS
from src.utils.config import get_dir

PROB_COLUMNS = [f"p_{i}" for i in LABEL_IDS]


def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_IDS, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_IDS, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_IDS, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(macro_p),
        "recall_macro": float(macro_r),
        "f1_macro": float(macro_f1),
        "f1_weighted": float(weighted_f1),
        "per_class": {
            ID2LABEL[i]: {
                "precision": float(precision[k]),
                "recall": float(recall[k]),
                "f1": float(f1[k]),
                "support": int(support[k]),
            }
            for k, i in enumerate(LABEL_IDS)
        },
        # rows = true class, columns = predicted class
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_IDS).tolist(),
    }


def save_results(
    cfg: dict,
    model_name: str,
    val_metrics: dict,
    test_metrics: dict,
    params: dict,
    extra: dict | None = None,
) -> Path:
    """Write reports/metrics/<model>.json - the single source of truth for comparisons."""
    payload = {
        "model": model_name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": params,
        "validation": val_metrics,
        "test": test_metrics,
    }
    payload.update(extra or {})
    path = get_dir(cfg, "metrics_dir") / f"{model_name}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def load_results(cfg: dict) -> list[dict]:
    results = []
    for path in sorted(get_dir(cfg, "metrics_dir").glob("*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # The folder also holds <model>_history.json (a list of epochs); only
        # result files written by save_results() belong in the comparison.
        if isinstance(data, dict) and "test" in data and "validation" in data:
            results.append(data)
    return results


def save_predictions(cfg: dict, model_name: str, texts, y_true, probs: np.ndarray) -> Path:
    """Save test-set predictions (used by error analysis, bias audit, robustness)."""
    frame = pd.DataFrame({"text": list(texts), "y_true": np.asarray(y_true)})
    frame["y_pred"] = np.asarray(probs).argmax(axis=1)
    for k, col in enumerate(PROB_COLUMNS):
        frame[col] = np.asarray(probs)[:, k]
    path = get_dir(cfg, "predictions_dir") / f"{model_name}_test.csv"
    frame.to_csv(path, index=False, encoding="utf-8")
    return path


def load_predictions(cfg: dict, model_name: str) -> pd.DataFrame:
    path = get_dir(cfg, "predictions_dir") / f"{model_name}_test.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Train '{model_name}' first.")
    return pd.read_csv(path, keep_default_na=False, encoding="utf-8")
