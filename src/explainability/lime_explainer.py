"""Explainability with LIME, plus exact global weights for the linear baseline.

    python -m src.explainability.lime_explainer --model bert --num-examples 10

Why LIME and not SHAP
---------------------
Both attribute a prediction to input words. LIME is model-agnostic through a
single ``predict_proba(texts)`` function, so the SAME explainer works for
TF-IDF models, the CNN/BiLSTM and BERT, and it is light enough to run on a CPU
inside the Streamlit app. SHAP's transformer explainer is slower and heavier.

How LIME works: it creates ~1000 variants of the text with random words removed,
asks the model for probabilities on each, and fits a small weighted linear model
to those outputs. The linear model's coefficients are the word weights.

What an explanation is NOT
--------------------------
* It is a local, approximate surrogate - it describes how the output changes
  when words are deleted, not the model's internal reasoning.
* Weights vary between runs (random sampling) and assume words act independently,
  which is false for a contextual model such as BERT.
* A word with a high weight is evidence of what the model reacts to, not proof
  that the prediction is right.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from src.constants import ID2LABEL, LABEL_NAMES
from src.utils.config import get_dir, load_config
from src.utils.logger import get_logger

log = get_logger(__name__)


def explain_prediction(predictor, text: str, num_features: int = 10, num_samples: int = 1000, seed: int = 42) -> dict:
    """Return LIME word weights for the predicted class of ``text``."""
    from lime.lime_text import LimeTextExplainer

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text is empty.")
    probs = predictor.predict_proba([text])[0]
    label_id = int(np.argmax(probs))
    explainer = LimeTextExplainer(class_names=LABEL_NAMES, random_state=seed)
    explanation = explainer.explain_instance(
        text, predictor.predict_proba, labels=[label_id], num_features=num_features, num_samples=num_samples
    )
    return {
        "label_id": label_id,
        "label": ID2LABEL[label_id],
        "probabilities": {ID2LABEL[i]: float(probs[i]) for i in ID2LABEL},
        # (word, weight): positive pushes TOWARDS the predicted class, negative away from it
        "weights": [(str(word), float(weight)) for word, weight in explanation.as_list(label=label_id)],
    }


def global_linear_importance(pipeline, top_k: int = 20) -> dict[str, list[tuple[str, float]]]:
    """Top n-grams per class from a linear model's coefficients (exact, not approximate).

    Works for the logistic-regression pipeline. For a linear model the
    coefficient IS the feature's contribution per unit of TF-IDF, so this is a
    faithful global explanation - a useful sanity check against LIME.
    """
    classifier = pipeline.named_steps["clf"]
    if not hasattr(classifier, "coef_"):
        raise ValueError("Global importance needs a linear model with coef_ (use logistic_regression).")
    features = pipeline.named_steps["tfidf"].get_feature_names_out()
    result = {}
    for row, class_id in zip(classifier.coef_, classifier.classes_):
        top = np.argsort(row)[::-1][:top_k]
        result[ID2LABEL[int(class_id)]] = [(str(features[i]), float(row[i])) for i in top]
    return result


def main() -> int:
    from src.evaluation.metrics import PROB_COLUMNS, load_predictions
    from src.inference.predictor import ModelNotFoundError, Predictor
    from src.preprocessing.text_cleaner import light_clean

    parser = argparse.ArgumentParser(description="Write LIME explanations for the most confident test errors")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-examples", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config()
    try:
        predictor = Predictor(args.model, cfg)
        predictions = load_predictions(cfg, args.model)
    except (ModelNotFoundError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 1

    exp_cfg = cfg["explainability"]
    errors = predictions[predictions["y_true"] != predictions["y_pred"]].copy()
    errors["confidence"] = errors[PROB_COLUMNS].max(axis=1)
    errors = errors.sort_values("confidence", ascending=False).head(args.num_examples)

    lines = [
        f"# LIME explanations - {args.model}",
        "",
        "> Content warning: contains offensive text from the dataset. Explanations are approximate.",
        "",
    ]
    for _, row in errors.iterrows():
        result = explain_prediction(predictor, row["text"], exp_cfg["lime_num_features"],
                                    exp_cfg["lime_num_samples"], cfg["project"]["seed"])
        lines += [
            f"**True:** {ID2LABEL[int(row['y_true'])]} | **Predicted:** {result['label']} "
            f"({result['probabilities'][result['label']]:.2f})",
            "",
            f"Text: `{light_clean(row['text'])[:240]}`",
            "",
            "| Word | Weight towards predicted class |",
            "|---|---:|",
        ]
        lines += [f"| {word} | {weight:+.3f} |" for word, weight in result["weights"]]
        lines.append("")

    if predictor.kind == "sklearn":
        try:
            importance = global_linear_importance(predictor._pipeline)
            lines += ["## Global n-gram weights (exact, from model coefficients)", ""]
            for label, items in importance.items():
                lines += [f"**{label}:** " + ", ".join(f"{w} ({c:+.2f})" for w, c in items), ""]
        except ValueError:
            pass

    out_dir = get_dir(cfg, "reports_dir") / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_lime.md"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
