"""Error analysis from saved test predictions.

    python -m src.evaluation.error_analysis --model bert

Writes reports/error_analysis/<model>.md (git-ignored: it quotes dataset text).

The safety-critical view is the hate-speech class:
* false negative - hate speech the model let through (harm to targets)
* false positive - non-hate text flagged as hate (harm to the author:
  wrongful removal, and it falls hardest on dialects and reclaimed in-group terms)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src.constants import ID2LABEL, LABEL_IDS
from src.evaluation.metrics import PROB_COLUMNS, load_predictions
from src.preprocessing.text_cleaner import light_clean
from src.utils.config import get_dir, load_config
from src.utils.logger import get_logger

log = get_logger(__name__)
HATE = 0


def summarise_errors(predictions: pd.DataFrame) -> dict:
    """Aggregate error statistics (no text)."""
    y_true, y_pred = predictions["y_true"].to_numpy(), predictions["y_pred"].to_numpy()
    wrong = y_true != y_pred
    pairs = {}
    for true_id in LABEL_IDS:
        for pred_id in LABEL_IDS:
            if true_id != pred_id:
                pairs[f"{ID2LABEL[true_id]} -> {ID2LABEL[pred_id]}"] = int(((y_true == true_id) & (y_pred == pred_id)).sum())

    words = predictions["text"].str.split().str.len()
    buckets = pd.cut(words, bins=[0, 8, 16, 24, np.inf], labels=["1-8", "9-16", "17-24", "25+"])
    by_length = {
        str(bucket): {"n": int(mask.sum()), "error_rate": round(float(wrong[mask.to_numpy()].mean()), 4)}
        for bucket in buckets.cat.categories
        if (mask := buckets == bucket).any()
    }
    confidence = predictions[PROB_COLUMNS].max(axis=1).to_numpy()
    return {
        "n": int(len(predictions)),
        "errors": int(wrong.sum()),
        "error_rate": round(float(wrong.mean()), 4),
        "confusion_pairs": dict(sorted(pairs.items(), key=lambda kv: -kv[1])),
        "hate_false_negatives": int(((y_true == HATE) & (y_pred != HATE)).sum()),
        "hate_false_positives": int(((y_true != HATE) & (y_pred == HATE)).sum()),
        "hate_support": int((y_true == HATE).sum()),
        "error_rate_by_length_words": by_length,
        "mean_confidence_correct": round(float(confidence[~wrong].mean()), 4) if (~wrong).any() else None,
        "mean_confidence_wrong": round(float(confidence[wrong].mean()), 4) if wrong.any() else None,
    }


def _examples(frame: pd.DataFrame, limit: int) -> list[str]:
    frame = frame.assign(confidence=frame[PROB_COLUMNS].max(axis=1)).sort_values("confidence", ascending=False)
    lines = ["| True | Predicted | Confidence | Text (handles masked) |", "|---|---|---:|---|"]
    for _, row in frame.head(limit).iterrows():
        text = light_clean(row["text"])[:220].replace("|", "/").replace("\n", " ")
        lines.append(f"| {ID2LABEL[int(row['y_true'])]} | {ID2LABEL[int(row['y_pred'])]} | {row['confidence']:.2f} | {text} |")
    return lines


def write_report(model_name: str, predictions: pd.DataFrame, path, limit: int = 15) -> dict:
    summary = summarise_errors(predictions)
    y_true, y_pred = predictions["y_true"], predictions["y_pred"]
    lines = [
        f"# Error analysis - {model_name}",
        "",
        "> Content warning: this file quotes offensive and hateful text from the dataset.",
        "",
        f"- Test tweets: {summary['n']}, errors: {summary['errors']} ({100 * summary['error_rate']:.1f}%)",
        f"- Hate speech: {summary['hate_false_negatives']} false negatives out of {summary['hate_support']} "
        f"true hate tweets; {summary['hate_false_positives']} false positives",
        f"- Mean confidence when correct: {summary['mean_confidence_correct']}, when wrong: {summary['mean_confidence_wrong']}",
        "",
        "## Confusion pairs (true -> predicted)",
        "",
    ]
    lines += [f"- {pair}: {count}" for pair, count in summary["confusion_pairs"].items()]
    lines += ["", "## Error rate by tweet length (words)", ""]
    lines += [f"- {bucket}: {v['error_rate']:.3f} (n={v['n']})" for bucket, v in summary["error_rate_by_length_words"].items()]
    lines += ["", "## Most confident FALSE NEGATIVES for hate speech", ""]
    lines += _examples(predictions[(y_true == HATE) & (y_pred != HATE)], limit)
    lines += ["", "## Most confident FALSE POSITIVES for hate speech", ""]
    lines += _examples(predictions[(y_true != HATE) & (y_pred == HATE)], limit)
    lines += ["", "## Other most confident errors", ""]
    lines += _examples(predictions[(y_true != y_pred) & (y_true != HATE) & (y_pred != HATE)], limit)
    lines += [
        "",
        "## Questions to answer when reading the examples",
        "",
        "1. Is the gold label itself debatable? (Crowd labels disagree often on hate vs offensive.)",
        "2. Does the error hinge on one trigger word regardless of who is speaking or why?",
        "3. Is it quoted, reclaimed, sarcastic or counter-speech? Those need context the tweet alone lacks.",
        "4. Does the text mention an identity group without attacking it?",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Write an error analysis report for one model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()
    cfg = load_config()
    try:
        predictions = load_predictions(cfg, args.model)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    out_dir = get_dir(cfg, "reports_dir") / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = write_report(args.model, predictions, out_dir / f"{args.model}.md", args.limit)
    log.info("Wrote %s | %s", out_dir / f"{args.model}.md", {k: summary[k] for k in ("errors", "error_rate")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
