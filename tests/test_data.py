import pandas as pd
import pytest

from src.data.eda import compute_eda
from src.data.load import DataValidationError, load_raw
from src.data.split import clean_and_deduplicate, load_splits, stratified_split


def _write_raw(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)          # writes an unnamed index column, like the real file
    return path


def _raw_like(frame):
    return pd.DataFrame({
        "count": 3, "hate_speech": (frame["label"] == 0) * 3, "offensive_language": (frame["label"] == 1) * 3,
        "neither": (frame["label"] == 2) * 3, "class": frame["label"], "tweet": frame["text"],
    })


def test_load_raw_standardises_columns(tmp_path, synthetic_frame):
    df = load_raw(_write_raw(tmp_path / "raw.csv", _raw_like(synthetic_frame)))
    assert {"text", "label", "count"} <= set(df.columns)
    assert not any(str(c).startswith("Unnamed") for c in df.columns)
    assert len(df) == len(synthetic_frame)


def test_load_raw_accepts_hugging_face_column_names(tmp_path, synthetic_frame):
    hf = _raw_like(synthetic_frame).rename(columns={"hate_speech": "hate_speech_count"})
    assert "hate_speech" in load_raw(_write_raw(tmp_path / "hf.csv", hf)).columns


def test_load_raw_rejects_wrong_schema_and_labels(tmp_path):
    with pytest.raises(DataValidationError):
        load_raw(_write_raw(tmp_path / "bad.csv", pd.DataFrame({"foo": [1], "bar": ["x"]})))
    with pytest.raises(DataValidationError):
        load_raw(_write_raw(tmp_path / "bad2.csv", pd.DataFrame({"class": [7], "tweet": ["x"]})))
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "missing.csv")


def test_deduplication_ignores_handles_and_drops_empty():
    df = pd.DataFrame({
        "text": ["RT @a: same words here", "RT @b: same words here", "different one", "!!!"],
        "label": [1, 1, 2, 2],
    })
    cleaned, stats = clean_and_deduplicate(df)
    assert len(cleaned) == 2
    assert stats["rows_duplicate"] == 1 and stats["rows_empty_after_cleaning"] == 1


def test_split_is_stratified_and_disjoint(synthetic_frame):
    cleaned, _ = clean_and_deduplicate(synthetic_frame)
    train, val, test = stratified_split(cleaned, 0.2, 0.2, seed=1)
    assert len(train) + len(val) + len(test) == len(cleaned)
    for part in (train, val, test):
        shares = part["label"].value_counts(normalize=True)
        assert all(abs(share - 1 / 3) < 0.05 for share in shares)
    texts = [set(part["text_clean"]) for part in (train, val, test)]
    assert not (texts[0] & texts[1]) and not (texts[0] & texts[2]) and not (texts[1] & texts[2])


def test_split_is_reproducible(synthetic_frame):
    cleaned, _ = clean_and_deduplicate(synthetic_frame)
    first = stratified_split(cleaned, 0.2, 0.2, seed=3)[2]["text"].tolist()
    second = stratified_split(cleaned, 0.2, 0.2, seed=3)[2]["text"].tolist()
    assert first == second


def test_split_rejects_bad_sizes(synthetic_frame):
    with pytest.raises(ValueError):
        stratified_split(synthetic_frame, 0.6, 0.5, seed=0)


def test_load_splits_reports_missing_files(tmp_cfg):
    with pytest.raises(FileNotFoundError):
        load_splits(tmp_cfg)


def test_load_splits_roundtrip(prepared_cfg):
    splits = load_splits(prepared_cfg)
    assert set(splits) == {"train", "val", "test"}
    assert splits["train"]["text_clean"].map(type).eq(str).all()


def test_compute_eda(synthetic_frame):
    summary = compute_eda(_raw_like(synthetic_frame).rename(columns={"class": "label", "tweet": "text"}))
    assert summary["rows"] == len(synthetic_frame)
    assert sum(summary["class_counts"].values()) == len(synthetic_frame)
    assert summary["pct_unanimous"] == 100.0
