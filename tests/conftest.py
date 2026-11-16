"""Shared fixtures.

The corpus below is SYNTHETIC and deliberately meaningless (colour/tree/weather
words). It exists only to exercise the code paths quickly; it is never used to
train or evaluate a real model, and no result from it is reported anywhere.
"""
from __future__ import annotations

import copy
import random

import pandas as pd
import pytest

from src.data.split import OUTPUT_COLUMNS, clean_and_deduplicate, stratified_split
from src.utils.config import load_config

_CLASS_WORDS = {
    0: ["amber", "arrow", "anchor", "apple"],
    1: ["birch", "bloom", "breeze", "bridge"],
    2: ["cedar", "coral", "cloud", "candle"],
}
_FILLER = ["the", "a", "today", "really", "with", "and", "near", "over"]


def make_synthetic_frame(per_class: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for label, words in _CLASS_WORDS.items():
        for i in range(per_class):
            tokens = rng.choices(words, k=4) + rng.choices(_FILLER, k=3) + [f"item{label}x{i}"]
            rng.shuffle(tokens)
            rows.append({"text": " ".join(tokens), "label": label})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.fixture()
def synthetic_frame() -> pd.DataFrame:
    return make_synthetic_frame()


@pytest.fixture()
def tmp_cfg(tmp_path) -> dict:
    """Real config, but every path points into pytest's temp dir and tracking is off."""
    cfg = copy.deepcopy(load_config())
    cfg["paths"] = {
        "raw_data": str(tmp_path / "data" / "raw" / "labeled_data.csv"),
        "processed_dir": str(tmp_path / "data" / "processed"),
        "models_dir": str(tmp_path / "models"),
        "reports_dir": str(tmp_path / "reports"),
        "figures_dir": str(tmp_path / "reports" / "figures"),
        "metrics_dir": str(tmp_path / "reports" / "metrics"),
        "predictions_dir": str(tmp_path / "reports" / "predictions"),
        "logs_dir": str(tmp_path / "logs"),
    }
    cfg["tracking"]["use_mlflow"] = False
    cfg["baselines"]["tfidf"]["min_df"] = 1
    cfg["baselines"]["grids"] = {"logistic_regression": [1.0], "linear_svm": [1.0], "naive_bayes": [1.0]}
    cfg["neural"].update({"epochs": 2, "batch_size": 16, "min_freq": 1, "embed_dim": 16, "max_len": 12})
    cfg["neural"]["cnn"]["num_filters"] = 8
    cfg["neural"]["bilstm"]["hidden_dim"] = 8
    cfg["robustness"]["sample_size"] = 30
    return cfg


@pytest.fixture()
def prepared_cfg(tmp_cfg, synthetic_frame) -> dict:
    """tmp_cfg with train/val/test CSVs already written."""
    from src.utils.config import get_dir

    cleaned, _ = clean_and_deduplicate(synthetic_frame)
    train, val, test = stratified_split(cleaned, 0.2, 0.2, seed=0)
    out_dir = get_dir(tmp_cfg, "processed_dir")
    for name, frame in {"train": train, "val": val, "test": test}.items():
        frame[OUTPUT_COLUMNS].to_csv(out_dir / f"{name}.csv", index=False)
    return tmp_cfg
