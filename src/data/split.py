"""Clean, de-duplicate and split the dataset ONCE, then save the splits.

    python -m src.data.split

Why splits are written to disk: every model (TF-IDF, CNN, BiLSTM, BERT) must be
trained and tested on exactly the same rows, otherwise the comparison is unfair.

Leakage controls
----------------
1. De-duplication happens BEFORE splitting. Twitter data is full of retweets;
   if the same text landed in train and test, test scores would be inflated.
   Duplicates are detected on the cleaned text, so "RT @a: xyz" and
   "RT @b: xyz" count as the same tweet.
2. The split is stratified: each split keeps the original class proportions,
   which matters when one class is only ~6% of the data.
3. Nothing is *fitted* here. TF-IDF, the vocabulary and all model weights are
   fitted later on the training split only.
4. The test split is touched once per model, after model selection on validation.
"""
from __future__ import annotations

import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load import load_raw
from src.preprocessing.text_cleaner import full_clean, light_clean
from src.utils.config import get_dir, load_config, resolve_path
from src.utils.logger import get_logger

log = get_logger(__name__)

SPLIT_NAMES = ("train", "val", "test")
OUTPUT_COLUMNS = ["text", "text_light", "text_clean", "label"]


def clean_and_deduplicate(df: pd.DataFrame, drop_duplicates: bool = True) -> tuple[pd.DataFrame, dict]:
    """Add the two cleaned text columns, drop empty rows and exact duplicates."""
    out = df.copy()
    out["text_light"] = out["text"].map(light_clean)
    out["text_clean"] = out["text"].map(full_clean)

    stats = {"rows_raw": int(len(out))}
    empty = out["text_clean"].str.len() == 0
    stats["rows_empty_after_cleaning"] = int(empty.sum())
    out = out[~empty]

    duplicated = out.duplicated(subset="text_clean", keep="first")
    stats["rows_duplicate"] = int(duplicated.sum())
    # Same text labelled differently = annotation noise worth knowing about.
    labels_per_text = out.groupby("text_clean")["label"].nunique()
    stats["texts_with_conflicting_labels"] = int((labels_per_text > 1).sum())
    if drop_duplicates:
        out = out[~duplicated]
    stats["rows_final"] = int(len(out))
    return out.reset_index(drop=True), stats


def stratified_split(
    df: pd.DataFrame, val_size: float, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split. ``val_size``/``test_size`` are fractions of the whole."""
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be in (0, 1) and sum to less than 1")
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val["label"], random_state=seed
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def prepare_splits(cfg: dict) -> dict[str, pd.DataFrame]:
    raw = load_raw(resolve_path(cfg["paths"]["raw_data"]))
    cleaned, stats = clean_and_deduplicate(raw, cfg["data"].get("drop_duplicates", True))
    log.info("Cleaning stats: %s", stats)

    train, val, test = stratified_split(
        cleaned, cfg["data"]["val_size"], cfg["data"]["test_size"], cfg["project"]["seed"]
    )
    out_dir = get_dir(cfg, "processed_dir")
    splits = {"train": train, "val": val, "test": test}
    for name, frame in splits.items():
        frame[OUTPUT_COLUMNS].to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
        shares = frame["label"].value_counts(normalize=True).sort_index().round(3).to_dict()
        log.info("%-5s %6d rows | class shares %s", name, len(frame), shares)
    return splits


def load_splits(cfg: dict) -> dict[str, pd.DataFrame]:
    """Read the saved splits. All training/evaluation scripts go through here."""
    processed = resolve_path(cfg["paths"]["processed_dir"])
    splits = {}
    for name in SPLIT_NAMES:
        path = processed / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Run:  python -m src.data.download  then  python -m src.data.split"
            )
        # keep_default_na=False: a tweet that literally says "null" or "NA" must stay a string.
        frame = pd.read_csv(path, keep_default_na=False, encoding="utf-8")
        frame["label"] = frame["label"].astype(int)
        splits[name] = frame
    return splits


if __name__ == "__main__":
    try:
        prepare_splits(load_config())
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)
