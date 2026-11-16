"""End-to-end test of the classical pipeline on the synthetic corpus:
train -> save -> metrics/predictions -> Predictor -> comparison -> robustness -> reports."""
import json

import numpy as np
import pytest

from src.constants import BASELINE_MODELS
from src.data.split import load_splits
from src.evaluation.error_analysis import write_report
from src.evaluation.evaluate import build_comparison, to_markdown
from src.evaluation.metrics import load_predictions, load_results
from src.evaluation.robustness import run_robustness
from src.explainability.lime_explainer import global_linear_importance
from src.inference.predictor import ModelNotFoundError, Predictor, available_models
from src.models.baselines import build_pipeline
from src.training.train_baselines import train_one
from src.utils.config import resolve_path


@pytest.fixture()
def trained_cfg(prepared_cfg):
    splits = load_splits(prepared_cfg)
    for name in BASELINE_MODELS:
        train_one(name, splits, prepared_cfg)
    return prepared_cfg


def test_unknown_baseline_raises():
    with pytest.raises(ValueError):
        build_pipeline("random_forest", {}, 1.0)


def test_training_writes_artifacts(trained_cfg):
    models_dir = resolve_path(trained_cfg["paths"]["models_dir"])
    metrics_dir = resolve_path(trained_cfg["paths"]["metrics_dir"])
    for name in BASELINE_MODELS:
        assert (models_dir / f"{name}.joblib").is_file()
        with open(metrics_dir / f"{name}.json", encoding="utf-8") as fh:
            payload = json.load(fh)
        assert {"validation", "test", "params", "search"} <= set(payload)
        # The synthetic classes are trivially separable; this checks wiring, not model quality.
        assert payload["test"]["f1_macro"] > 0.9
        predictions = load_predictions(trained_cfg, name)
        assert np.allclose(predictions[["p_0", "p_1", "p_2"]].sum(axis=1), 1.0, atol=1e-6)
    assert (resolve_path(trained_cfg["paths"]["figures_dir"]) / "confusion_logistic_regression.png").is_file()


def test_predictor_serves_every_baseline(trained_cfg):
    assert set(available_models(trained_cfg)) == set(BASELINE_MODELS)
    for name in BASELINE_MODELS:
        predictor = Predictor(name, trained_cfg)
        result = predictor.predict("cedar coral cloud candle")
        assert result["label"] == "Neither"
        assert sum(result["probabilities"].values()) == pytest.approx(1.0)
        assert predictor.predict_proba(["amber arrow", "birch bloom"]).shape == (2, 3)


def test_predictor_errors(trained_cfg):
    with pytest.raises(ModelNotFoundError):
        Predictor("does_not_exist", trained_cfg)
    with pytest.raises(ValueError):
        Predictor("naive_bayes", trained_cfg).predict("   ")


def test_comparison_table(trained_cfg):
    table = build_comparison(load_results(trained_cfg))
    assert len(table) == 3 and table["f1_macro"].is_monotonic_decreasing
    assert "f1_hate_speech" in to_markdown(table)


def test_robustness_and_reports(trained_cfg, tmp_path):
    predictor = Predictor("logistic_regression", trained_cfg)
    test = load_splits(trained_cfg)["test"]
    rows = run_robustness(predictor, test["text"].tolist(), test["label"].to_numpy(), seed=0)
    assert rows[0]["perturbation"] == "clean" and len(rows) == 6
    assert all(0.0 <= r["flip_rate"] <= 1.0 for r in rows)

    summary = write_report("logistic_regression", load_predictions(trained_cfg, "logistic_regression"),
                           tmp_path / "errors.md")
    assert (tmp_path / "errors.md").is_file() and summary["n"] == len(test)

    importance = global_linear_importance(predictor._pipeline, top_k=3)
    assert importance["Neither"][0][0] in {"cedar", "coral", "cloud", "candle"}


def test_lime_explanation(trained_cfg):
    pytest.importorskip("lime")
    from src.explainability.lime_explainer import explain_prediction

    predictor = Predictor("logistic_regression", trained_cfg)
    result = explain_prediction(predictor, "cedar coral cloud today", num_features=4, num_samples=200)
    assert result["label"] == "Neither" and len(result["weights"]) > 0
