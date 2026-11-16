"""Phase 3 - train the TF-IDF baselines.

    python -m src.training.train_baselines
    python -m src.training.train_baselines --models logistic_regression

Protocol per model:
  1. for each hyper-parameter value: fit on TRAIN, score macro-F1 on VALIDATION
  2. keep the best value (model selection never sees the test set)
  3. evaluate that single model once on TEST
"""
from __future__ import annotations

import argparse
import sys

import joblib

from src.constants import BASELINE_MODELS
from src.data.split import load_splits
from src.evaluation.metrics import compute_metrics, save_predictions, save_results
from src.evaluation.plots import plot_confusion_matrix
from src.models.baselines import PARAM_NAMES, build_pipeline
from src.utils.config import get_dir, load_config
from src.utils.logger import get_logger
from src.utils.runtime import set_seed
from src.utils.tracking import tracked_run

log = get_logger(__name__)


def train_one(model_name: str, splits: dict, cfg: dict) -> dict:
    seed = cfg["project"]["seed"]
    tfidf_cfg = cfg["baselines"]["tfidf"]
    grid = cfg["baselines"]["grids"][model_name]
    train, val, test = splits["train"], splits["val"], splits["test"]

    best = {"score": -1.0, "param": None, "pipeline": None}
    search_log = []
    for param in grid:
        pipeline = build_pipeline(model_name, tfidf_cfg, param, seed)
        pipeline.fit(train["text_clean"], train["label"])
        val_f1 = compute_metrics(val["label"], pipeline.predict(val["text_clean"]))["f1_macro"]
        search_log.append({PARAM_NAMES[model_name]: param, "val_f1_macro": round(val_f1, 4)})
        log.info("%s %s=%s -> val macro-F1 %.4f", model_name, PARAM_NAMES[model_name], param, val_f1)
        if val_f1 > best["score"]:
            best = {"score": val_f1, "param": param, "pipeline": pipeline}

    pipeline = best["pipeline"]
    val_metrics = compute_metrics(val["label"], pipeline.predict(val["text_clean"]))
    test_probs = pipeline.predict_proba(test["text_clean"])
    test_metrics = compute_metrics(test["label"], test_probs.argmax(axis=1))

    params = {PARAM_NAMES[model_name]: best["param"], **tfidf_cfg}
    with tracked_run(cfg, model_name, params) as tracker:
        tracker.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        tracker.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    model_path = get_dir(cfg, "models_dir") / f"{model_name}.joblib"
    joblib.dump(pipeline, model_path)
    save_results(cfg, model_name, val_metrics, test_metrics, params, {"search": search_log})
    save_predictions(cfg, model_name, test["text"], test["label"], test_probs)
    plot_confusion_matrix(
        test_metrics["confusion_matrix"], model_name, get_dir(cfg, "figures_dir") / f"confusion_{model_name}.png"
    )
    log.info(
        "%s | best %s=%s | TEST macro-F1 %.4f, accuracy %.4f | saved %s",
        model_name, PARAM_NAMES[model_name], best["param"],
        test_metrics["f1_macro"], test_metrics["accuracy"], model_path,
    )
    return test_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train TF-IDF baselines")
    parser.add_argument("--models", nargs="+", default=BASELINE_MODELS, choices=BASELINE_MODELS)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    try:
        splits = load_splits(cfg)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    for model_name in args.models:
        train_one(model_name, splits, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
