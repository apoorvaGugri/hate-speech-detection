"""Load and validate the raw dataset.

Raw schema (GitHub CSV):
    <index>, count, hate_speech, offensive_language, neither, class, tweet

* count               - number of CrowdFlower annotators for the tweet (>= 3)
* hate_speech         - how many of them chose "hate speech"
* offensive_language  - how many chose "offensive language"
* neither             - how many chose "neither"
* class               - majority label: 0 hate speech, 1 offensive language, 2 neither
* tweet               - the tweet text

The Hugging Face mirror uses ``*_count`` column names; both are accepted and
mapped onto one internal schema with ``text`` and ``label`` columns.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.constants import LABEL_IDS

_RENAME = {
    "class": "label",
    "tweet": "text",
    "hate_speech_count": "hate_speech",
    "offensive_language_count": "offensive_language",
    "neither_count": "neither",
}
VOTE_COLUMNS = ["hate_speech", "offensive_language", "neither"]


class DataValidationError(Exception):
    """Raised when a file does not look like the Davidson dataset."""


def load_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run:  python -m src.data.download"
        )
    try:
        df = pd.read_csv(path) if path.suffix.lower() != ".parquet" else pd.read_parquet(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        raise DataValidationError(f"Could not parse {path}: {exc}") from exc

    df = df.rename(columns=_RENAME)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])

    missing = [c for c in ("text", "label") if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required column(s) {missing}; found {list(df.columns)}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    try:
        df["label"] = df["label"].astype(int)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"Label column is not integer-valued: {exc}") from exc

    unexpected = sorted(set(df["label"].unique()) - set(LABEL_IDS))
    if unexpected:
        raise DataValidationError(f"Unexpected label values {unexpected}; expected {LABEL_IDS}")
    if df.empty:
        raise DataValidationError("Dataset contains no usable rows")
    return df.reset_index(drop=True)
