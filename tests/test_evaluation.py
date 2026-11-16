import random

import numpy as np
import pandas as pd
import pytest

from src.evaluation.bias_audit import subgroup_rates
from src.evaluation.error_analysis import summarise_errors
from src.evaluation.metrics import compute_metrics, load_results, save_results
from src.evaluation.robustness import PERTURBATIONS, perturb_all
from src.training.common import balanced_class_weights


def test_metrics_perfect_prediction():
    metrics = compute_metrics([0, 1, 2, 1], [0, 1, 2, 1])
    assert metrics["accuracy"] == 1.0 and metrics["f1_macro"] == 1.0
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 2, 0], [0, 0, 1]]


def test_macro_f1_punishes_ignoring_the_minority_class():
    # 8 of 10 are class 1; predicting class 1 always gives 80% accuracy...
    y_true = [0, 1, 1, 1, 1, 1, 1, 1, 1, 2]
    metrics = compute_metrics(y_true, [1] * 10)
    assert metrics["accuracy"] == pytest.approx(0.8)
    # ...but macro-F1 exposes that two of three classes are never found.
    assert metrics["f1_macro"] == pytest.approx((0 + 16 / 18 + 0) / 3)
    assert metrics["f1_weighted"] > metrics["f1_macro"]
    assert metrics["per_class"]["Hate speech"]["recall"] == 0.0


def test_metrics_handle_a_missing_class():
    metrics = compute_metrics([1, 1, 2], [1, 2, 2])
    assert metrics["per_class"]["Hate speech"]["support"] == 0
    assert len(metrics["confusion_matrix"]) == 3


def test_balanced_class_weights_favour_rare_classes():
    weights = balanced_class_weights([0] * 10 + [1] * 80 + [2] * 10)
    assert weights[0] > weights[1] and weights[0] == pytest.approx(100 / 30)


def _predictions():
    return pd.DataFrame({
        "text": ["plain words here", "the women at work", "another plain one", "a muslim family", "short", "last text ok"],
        "y_true": [2, 2, 1, 2, 0, 0],
        "y_pred": [2, 0, 1, 2, 1, 0],
        "p_0": [0.1, 0.8, 0.1, 0.1, 0.2, 0.9],
        "p_1": [0.1, 0.1, 0.8, 0.1, 0.7, 0.05],
        "p_2": [0.8, 0.1, 0.1, 0.8, 0.1, 0.05],
    })


def test_error_summary_counts():
    summary = summarise_errors(_predictions())
    assert summary["errors"] == 2
    assert summary["hate_false_negatives"] == 1 and summary["hate_false_positives"] == 1
    assert summary["confusion_pairs"]["Neither -> Hate speech"] == 1


def test_subgroup_rates():
    rates = subgroup_rates(_predictions())
    assert rates["any_identity_term"]["tweets"] == 2
    assert rates["any_identity_term"]["neither_flagged"] == {"n": 2, "rate": 0.5}
    assert rates["no_identity_term"]["neither_flagged"] == {"n": 1, "rate": 0.0}


@pytest.mark.parametrize("name", list(PERTURBATIONS))
def test_perturbations_are_deterministic_strings(name):
    texts = ["this sentence contains several ordinary longer words", "ok"]
    first, second = perturb_all(texts, name, seed=5), perturb_all(texts, name, seed=5)
    assert first == second and all(isinstance(t, str) for t in first)


def test_perturbations_change_text():
    text = "this sentence contains several ordinary longer words " * 5
    rng = random.Random(0)
    assert all(fn(text, rng) != text for fn in PERTURBATIONS.values())
    assert np.char.isupper(PERTURBATIONS["uppercase"]("abc", rng))


def test_load_results_ignores_history_files(tmp_cfg):
    """Regression test: <model>_history.json lives in the same folder and is a list."""
    import json

    from src.utils.config import get_dir

    metrics = compute_metrics([0, 1, 2], [0, 1, 2])
    save_results(tmp_cfg, "cnn", metrics, metrics, {})
    with open(get_dir(tmp_cfg, "metrics_dir") / "cnn_history.json", "w", encoding="utf-8") as fh:
        json.dump([{"epoch": 1, "train_loss": 1.0}], fh)
    results = load_results(tmp_cfg)
    assert [r["model"] for r in results] == ["cnn"]
