"""Phase 6 - compare every trained model on the test set.

    python -m src.evaluation.evaluate

Reads reports/metrics/*.json (written by the training scripts) and produces
reports/model_comparison.csv, reports/model_comparison.md and
reports/figures/model_comparison.png. It reports only what was actually run:
a model that has not been trained simply does not appear.
"""
from __future__ import annotations

import sys

import pandas as pd

from src.evaluation.metrics import load_results
from src.evaluation.plots import plot_model_comparison
from src.utils.config import get_dir, load_config
from src.utils.logger import get_logger

log = get_logger(__name__)

COLUMNS = [
    "model", "f1_macro", "f1_weighted", "accuracy", "precision_macro", "recall_macro",
    "f1_hate_speech", "recall_hate_speech", "f1_offensive", "f1_neither", "val_f1_macro", "smoke_test",
]


def build_comparison(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        test, per_class = result["test"], result["test"]["per_class"]
        rows.append({
            "model": result["model"],
            "f1_macro": test["f1_macro"],
            "f1_weighted": test["f1_weighted"],
            "accuracy": test["accuracy"],
            "precision_macro": test["precision_macro"],
            "recall_macro": test["recall_macro"],
            "f1_hate_speech": per_class["Hate speech"]["f1"],
            "recall_hate_speech": per_class["Hate speech"]["recall"],
            "f1_offensive": per_class["Offensive language"]["f1"],
            "f1_neither": per_class["Neither"]["f1"],
            "val_f1_macro": result["validation"]["f1_macro"],
            "smoke_test": bool(result.get("smoke_test", False)),
        })
    table = pd.DataFrame(rows, columns=COLUMNS)
    return table.sort_values("f1_macro", ascending=False).reset_index(drop=True)


def to_markdown(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(table.columns) + " |"
    divider = "|" + "|".join(["---"] + ["---:"] * (len(table.columns) - 1)) + "|"
    lines = ["# Model comparison (test set)", "", "Sorted by macro-F1, the primary metric.", "", header, divider]
    for _, row in table.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    if table["smoke_test"].any():
        lines += ["", "Rows with smoke_test=True were trained on a subset and are NOT real results."]
    return "\n".join(lines) + "\n"


def main() -> int:
    cfg = load_config()
    results = load_results(cfg)
    if not results:
        log.error("No metrics found in %s. Train at least one model first.", get_dir(cfg, "metrics_dir"))
        return 1
    table = build_comparison(results)
    reports = get_dir(cfg, "reports_dir")
    table.to_csv(reports / "model_comparison.csv", index=False)
    with open(reports / "model_comparison.md", "w", encoding="utf-8") as fh:
        fh.write(to_markdown(table))
    plot_model_comparison(table, get_dir(cfg, "figures_dir") / "model_comparison.png")
    print(table.round(4).to_string(index=False))
    log.info("Comparison written to %s", reports)
    return 0


if __name__ == "__main__":
    sys.exit(main())
