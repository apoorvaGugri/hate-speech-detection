"""Unified inference: one ``Predictor`` interface for every trained model.

    from src.inference.predictor import Predictor
    predictor = Predictor("bert")
    predictor.predict("some text")

    python -m src.inference.predictor --model logistic_regression --text "have a nice day"

The predictor takes RAW text and applies the same cleaning used in training
(``full_clean`` for TF-IDF/CNN/BiLSTM, ``light_clean`` for transformers).
Train/serve skew - cleaning text differently at inference than at training -
is a classic silent bug, so the cleaning lives here and nowhere else.

Security note: ``.joblib`` files are pickles. Only load model files you trained yourself.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.constants import ID2LABEL, NUM_CLASSES
from src.preprocessing.text_cleaner import full_clean, light_clean
from src.utils.config import load_config, resolve_path

MAX_INPUT_CHARS = 5000


class ModelNotFoundError(FileNotFoundError):
    """Raised when no trained artefact exists for the requested model name."""


def available_models(cfg: dict | None = None) -> list[str]:
    """Names of trained models found in models/, transformers first."""
    cfg = cfg or load_config()
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    if not models_dir.is_dir():
        return []
    transformers_ = sorted(p.name for p in models_dir.iterdir() if p.is_dir() and (p / "config.json").is_file())
    neural = sorted(p.stem for p in models_dir.glob("*.pt"))
    classical = sorted(p.stem for p in models_dir.glob("*.joblib"))
    return transformers_ + neural + classical


class Predictor:
    def __init__(self, model_name: str, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.model_name = model_name
        models_dir = resolve_path(self.cfg["paths"]["models_dir"])
        joblib_path = models_dir / f"{model_name}.joblib"
        torch_path = models_dir / f"{model_name}.pt"
        hf_dir = models_dir / model_name

        if joblib_path.is_file():
            self.kind = "sklearn"
            self._load_sklearn(joblib_path)
        elif torch_path.is_file():
            self.kind = "neural"
            self._load_neural(torch_path)
        elif (hf_dir / "config.json").is_file():
            self.kind = "transformer"
            self._load_transformer(hf_dir)
        else:
            raise ModelNotFoundError(
                f"No trained model named '{model_name}' in {models_dir}. "
                f"Available: {available_models(self.cfg) or 'none - run a training script first'}"
            )

    # ------------------------------------------------------------------ loading
    def _load_sklearn(self, path: Path) -> None:
        import joblib

        self._pipeline = joblib.load(path)

    def _load_neural(self, path: Path) -> None:
        import torch

        from src.models.bilstm import BiLSTMClassifier
        from src.models.cnn import TextCNN
        from src.preprocessing.vocabulary import Vocabulary
        from src.utils.runtime import get_device

        self._device = get_device()
        # weights_only=True refuses arbitrary pickled objects (the checkpoint is plain tensors/dicts).
        checkpoint = torch.load(path, map_location=self._device, weights_only=True)
        model_cls = {"cnn": TextCNN, "bilstm": BiLSTMClassifier}[checkpoint["model_type"]]
        self._model = model_cls(**checkpoint["model_kwargs"])
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.to(self._device).eval()
        self._vocab = Vocabulary.from_dict(checkpoint["vocab"])
        self._max_len = int(checkpoint["max_len"])

    def _load_transformer(self, directory: Path) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from src.utils.runtime import get_device

        self._device = get_device()
        self._tokenizer = AutoTokenizer.from_pretrained(directory)
        self._model = AutoModelForSequenceClassification.from_pretrained(directory)
        self._model.to(self._device).eval()
        self._max_len = int(self.cfg["bert"]["max_len"])
        meta_path = directory / "training_meta.json"
        if meta_path.is_file():
            with open(meta_path, "r", encoding="utf-8") as fh:
                self._max_len = int(json.load(fh).get("max_len", self._max_len))

    # ---------------------------------------------------------------- inference
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Raw texts -> array of shape (n_texts, 3) with class probabilities."""
        if isinstance(texts, str):
            texts = [texts]
        texts = [str(t)[:MAX_INPUT_CHARS] for t in texts]
        if not texts:
            return np.zeros((0, NUM_CLASSES))
        if self.kind == "sklearn":
            return np.asarray(self._pipeline.predict_proba([full_clean(t) for t in texts]))
        if self.kind == "neural":
            return self._predict_neural([full_clean(t) for t in texts])
        return self._predict_transformer([light_clean(t) for t in texts])

    def _predict_neural(self, cleaned: list[str], batch_size: int = 256) -> np.ndarray:
        import torch

        outputs = []
        with torch.no_grad():
            for start in range(0, len(cleaned), batch_size):
                batch = cleaned[start:start + batch_size]
                ids = torch.tensor([self._vocab.encode(t, self._max_len) for t in batch],
                                   dtype=torch.long, device=self._device)
                outputs.append(torch.softmax(self._model(ids), dim=1).cpu().numpy())
        return np.concatenate(outputs)

    def _predict_transformer(self, cleaned: list[str], batch_size: int = 32) -> np.ndarray:
        import torch

        outputs = []
        with torch.no_grad():
            for start in range(0, len(cleaned), batch_size):
                batch = cleaned[start:start + batch_size]
                encoded = self._tokenizer(batch, padding=True, truncation=True,
                                          max_length=self._max_len, return_tensors="pt")
                encoded = {k: v.to(self._device) for k, v in encoded.items()}
                logits = self._model(**encoded).logits.float()
                outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(outputs)

    def predict(self, text: str) -> dict:
        """Classify one text. Raises ValueError on empty input."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text is empty.")
        probs = self.predict_proba([text])[0]
        label_id = int(np.argmax(probs))
        return {
            "label_id": label_id,
            "label": ID2LABEL[label_id],
            "confidence": float(probs[label_id]),
            "probabilities": {ID2LABEL[i]: float(probs[i]) for i in ID2LABEL},
            "model": self.model_name,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify one text from the command line")
    parser.add_argument("--model", default=None, help="model name (default: first available)")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    try:
        name = args.model or (available_models() or [None])[0]
        if name is None:
            raise ModelNotFoundError("No trained models found. Run a training script first.")
        print(json.dumps(Predictor(name).predict(args.text), indent=2))
    except (ModelNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
