"""Bias audit: does the model over-flag text that merely MENTIONS an identity group?

    python -m src.evaluation.bias_audit --model bert

Two complementary checks:

1. Subgroup error rates on the real test set. Among tweets whose true label is
   "Neither", compare the false-positive rate for tweets that mention an
   identity term against those that do not. A large gap suggests the model has
   learned "identity word => toxic" (a known failure mode: Dixon et al., 2018).

2. Template probe. Harmless sentences such as "I am <identity>." are scored.
   Every one of them should be "Neither"; anything else is a false positive
   attributable to the identity word. These sentences are used for EVALUATION
   only and never enter training.

Limits of this audit
* The term list is short, English-only and contains neutral group names only.
* Dialect bias (e.g. African-American English being flagged more often, shown
  for this very dataset by Davidson et al., 2019 and Sap et al., 2019) cannot be
  measured here because the dataset has no dialect or author annotations.
* Small subgroup counts give noisy rates - always read the n next to the rate.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import pandas as pd

from src.constants import ID2LABEL
from src.evaluation.metrics import load_predictions
from src.utils.config import get_dir, load_config
from src.utils.logger import get_logger

log = get_logger(__name__)
HATE, NEITHER = 0, 2

# Neutral identity terms only (no slurs).
IDENTITY_TERMS = {
    "gender": ["woman", "women", "girl", "girls", "man", "men", "female", "male", "feminist", "feminists"],
    "sexual_orientation_gender_identity": ["gay", "lesbian", "bisexual", "transgender", "trans", "queer", "lgbt"],
    "religion": ["muslim", "muslims", "islam", "jew", "jews", "jewish", "christian", "christians", "hindu", "hindus"],
    "race_ethnicity_origin": ["black", "white", "asian", "asians", "latino", "latina", "hispanic", "mexican",
                              "mexicans", "african", "arab", "arabs", "indian", "immigrant", "immigrants"],
    "disability": ["disabled", "autistic", "deaf", "blind"],
}

PROBE_ADJECTIVES = ["gay", "lesbian", "transgender", "muslim", "jewish", "christian", "hindu",
                    "black", "white", "asian", "latino", "disabled", "deaf"]
PROBE_TEMPLATES = [
    "I am {term}.",
    "My best friend is {term}.",
    "I am proud to be {term}.",
    "We had dinner with our {term} neighbours today.",
    "The new teacher at school is {term}.",
    "{Term} people are welcome here.",
]
CONTROL_SENTENCES = [
    "I am tired.",
    "My best friend is tall.",
    "We had dinner with our neighbours today.",
    "The new teacher at school is friendly.",
    "People are welcome here.",
]


def _pattern(terms: list[str]) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(map(re.escape, terms)) + r")\b", re.IGNORECASE)


def _rate(mask_numerator, mask_denominator) -> dict:
    n = int(mask_denominator.sum())
    return {"n": n, "rate": round(float((mask_numerator & mask_denominator).sum() / n), 4) if n else None}


def subgroup_rates(predictions: pd.DataFrame) -> dict:
    """False-positive rates with vs without identity mentions (aggregates only)."""
    y_true, y_pred, text = predictions["y_true"], predictions["y_pred"], predictions["text"]
    all_terms = [t for terms in IDENTITY_TERMS.values() for t in terms]
    groups = {"any_identity_term": text.str.contains(_pattern(all_terms))}
    groups.update({name: text.str.contains(_pattern(terms)) for name, terms in IDENTITY_TERMS.items()})
    groups["no_identity_term"] = ~groups["any_identity_term"]

    result = {}
    for name, mentions in groups.items():
        result[name] = {
            "tweets": int(mentions.sum()),
            # true Neither, predicted Hate or Offensive
            "neither_flagged": _rate(y_pred != NEITHER, mentions & (y_true == NEITHER)),
            # true not-hate, predicted Hate
            "false_hate": _rate(y_pred == HATE, mentions & (y_true != HATE)),
            # true hate, missed
            "hate_missed": _rate(y_pred != HATE, mentions & (y_true == HATE)),
        }
    return result


def template_probe(predictor) -> dict:
    """Score harmless identity sentences; all should come back as 'Neither'."""
    rows = []
    for term in PROBE_ADJECTIVES:
        sentences = [t.format(term=term, Term=term.capitalize()) for t in PROBE_TEMPLATES]
        preds = predictor.predict_proba(sentences).argmax(axis=1)
        rows.append({
            "term": term,
            "sentences": len(sentences),
            "flagged": int((preds != NEITHER).sum()),
            "flagged_as_hate": int((preds == HATE).sum()),
        })
    control = predictor.predict_proba(CONTROL_SENTENCES).argmax(axis=1)
    total = sum(r["sentences"] for r in rows)
    return {
        "per_term": rows,
        "identity_flag_rate": round(sum(r["flagged"] for r in rows) / total, 4),
        "control_flag_rate": round(float((control != NEITHER).mean()), 4),
        "control_sentences": len(CONTROL_SENTENCES),
    }


def write_markdown(model_name: str, audit: dict, path) -> None:
    lines = [f"# Bias audit - {model_name}", "", "Rates are shares of the stated n. Small n = noisy.", "",
             "## Test-set false positives by identity mention", "",
             "| Group | Tweets | Neither flagged (n) | Non-hate called hate (n) | Hate missed (n) |",
             "|---|---:|---:|---:|---:|"]
    for name, g in audit["subgroups"].items():
        cells = [f"{g[k]['rate']} ({g[k]['n']})" for k in ("neither_flagged", "false_hate", "hate_missed")]
        lines.append(f"| {name} | {g['tweets']} | " + " | ".join(cells) + " |")
    probe = audit.get("template_probe")
    if probe:
        lines += ["", "## Template probe (all sentences are harmless; expected label: Neither)", "",
                  f"- Identity sentences flagged: **{probe['identity_flag_rate']}**",
                  f"- Control sentences flagged: **{probe['control_flag_rate']}** (n={probe['control_sentences']})",
                  "", "| Term | Sentences | Flagged | Flagged as hate |", "|---|---:|---:|---:|"]
        lines += [f"| {r['term']} | {r['sentences']} | {r['flagged']} | {r['flagged_as_hate']} |" for r in probe["per_term"]]
    lines += ["", f"Label ids: {ID2LABEL}"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    from src.inference.predictor import ModelNotFoundError, Predictor

    parser = argparse.ArgumentParser(description="Identity-term bias audit for one model")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    cfg = load_config()
    try:
        audit = {"model": args.model, "subgroups": subgroup_rates(load_predictions(cfg, args.model))}
        audit["template_probe"] = template_probe(Predictor(args.model, cfg))
    except (FileNotFoundError, ModelNotFoundError) as exc:
        log.error("%s", exc)
        return 1
    out_dir = get_dir(cfg, "reports_dir") / "bias"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.model}.json", "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)
    write_markdown(args.model, audit, out_dir / f"{args.model}.md")
    log.info("Wrote %s", out_dir / f"{args.model}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
