"""Robustness testing: how much do predictions change under small input changes?

    python -m src.evaluation.robustness --model bert

People who want to evade a filter misspell words, swap letters for symbols or
pad a message with friendly text. Each perturbation below is applied to a
stratified sample of the test set and we report macro-F1, hate-speech recall
and the share of predictions that flip compared with the clean text.

The perturbations keep the meaning of the text for a human reader, so the gold
label is assumed unchanged.
"""
from __future__ import annotations

import argparse
import json
import random
import sys

import numpy as np
from sklearn.model_selection import train_test_split

from src.data.split import load_splits
from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import plot_robustness
from src.utils.config import get_dir, load_config
from src.utils.logger import get_logger

log = get_logger(__name__)

LEET = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "$"}


def _is_plain_word(word: str) -> bool:
    return word.isalpha() and len(word) > 3


def typo_swap(text: str, rng: random.Random, rate: float = 0.25) -> str:
    """Swap two neighbouring inner letters in about a quarter of the words."""
    words = text.split()
    for k, word in enumerate(words):
        if _is_plain_word(word) and rng.random() < rate:
            i = rng.randrange(1, len(word) - 2)
            words[k] = word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return " ".join(words)


def leetspeak(text: str, rng: random.Random, rate: float = 0.3) -> str:
    """Replace letters with look-alike symbols (a->@, e->3, ...) in ~30% of words."""
    words = text.split()
    for k, word in enumerate(words):
        if _is_plain_word(word) and rng.random() < rate:
            words[k] = "".join(LEET.get(ch.lower(), ch) for ch in word)
    return " ".join(words)


def char_separator(text: str, rng: random.Random, rate: float = 0.2) -> str:
    """Insert dots between the letters of ~20% of the words: w.o.r.d"""
    words = text.split()
    for k, word in enumerate(words):
        if _is_plain_word(word) and rng.random() < rate:
            words[k] = ".".join(word)
    return " ".join(words)


def uppercase(text: str, rng: random.Random) -> str:  # noqa: ARG001 - same signature as the others
    return text.upper()


def benign_suffix(text: str, rng: random.Random) -> str:  # noqa: ARG001
    """Append friendly words; a filter should not be talked out of a decision this way."""
    return text + " have a lovely day everyone, love and peace"


PERTURBATIONS = {
    "typo_swap": typo_swap,
    "leetspeak": leetspeak,
    "char_separator": char_separator,
    "uppercase": uppercase,
    "benign_suffix": benign_suffix,
}


def perturb_all(texts: list[str], name: str, seed: int) -> list[str]:
    rng = random.Random(seed)
    return [PERTURBATIONS[name](t, rng) for t in texts]


def run_robustness(predictor, texts: list[str], labels, seed: int) -> list[dict]:
    labels = np.asarray(labels)
    clean_pred = predictor.predict_proba(texts).argmax(axis=1)
    clean = compute_metrics(labels, clean_pred)
    rows = [{"perturbation": "clean", "f1_macro": clean["f1_macro"],
             "recall_hate_speech": clean["per_class"]["Hate speech"]["recall"], "flip_rate": 0.0}]
    for name in PERTURBATIONS:
        pred = predictor.predict_proba(perturb_all(texts, name, seed)).argmax(axis=1)
        metrics = compute_metrics(labels, pred)
        rows.append({
            "perturbation": name,
            "f1_macro": metrics["f1_macro"],
            "recall_hate_speech": metrics["per_class"]["Hate speech"]["recall"],
            "flip_rate": float((pred != clean_pred).mean()),
        })
    return rows


def main() -> int:
    from src.inference.predictor import ModelNotFoundError, Predictor

    parser = argparse.ArgumentParser(description="Perturbation robustness test for one model")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    cfg = load_config()
    seed = cfg["project"]["seed"]
    try:
        predictor = Predictor(args.model, cfg)
        test = load_splits(cfg)["test"]
    except (FileNotFoundError, ModelNotFoundError) as exc:
        log.error("%s", exc)
        return 1

    n = min(cfg["robustness"]["sample_size"], len(test))
    if n < len(test):
        test, _ = train_test_split(test, train_size=n, stratify=test["label"], random_state=seed)
    rows = run_robustness(predictor, test["text"].tolist(), test["label"].to_numpy(), seed)

    out_dir = get_dir(cfg, "reports_dir") / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.model}.json", "w", encoding="utf-8") as fh:
        json.dump({"model": args.model, "sample_size": n, "results": rows}, fh, indent=2)
    lines = [f"# Robustness - {args.model}", "", f"Stratified test sample: {n} tweets.", "",
             "| Perturbation | Macro-F1 | Hate-speech recall | Predictions flipped |", "|---|---:|---:|---:|"]
    lines += [f"| {r['perturbation']} | {r['f1_macro']:.4f} | {r['recall_hate_speech']:.4f} | {r['flip_rate']:.4f} |" for r in rows]
    with open(out_dir / f"{args.model}.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    plot_robustness(rows, f"Robustness - {args.model}", get_dir(cfg, "figures_dir") / f"robustness_{args.model}.png")
    for row in rows:
        print(f"{row['perturbation']:>15}: macro-F1 {row['f1_macro']:.4f} | flips {row['flip_rate']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
