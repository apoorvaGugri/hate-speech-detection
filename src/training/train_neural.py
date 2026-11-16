"""Phase 4 - train the CNN and the BiLSTM.

    python -m src.training.train_neural --model cnn
    python -m src.training.train_neural --model bilstm
    python -m src.training.train_neural --model cnn --epochs 2      # quick check

Both models learn their word embeddings from scratch (no GloVe download), run
comfortably on a CPU, and use:
  * a vocabulary built from the training split only,
  * class-weighted cross-entropy for the imbalance,
  * early stopping on validation macro-F1 (the best epoch's weights are kept).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.constants import NEURAL_MODELS, NUM_CLASSES
from src.data.split import load_splits
from src.evaluation.metrics import compute_metrics, save_predictions, save_results
from src.evaluation.plots import plot_confusion_matrix, plot_history
from src.models.bilstm import BiLSTMClassifier
from src.models.cnn import TextCNN
from src.preprocessing.vocabulary import PAD_ID, Vocabulary
from src.training.common import balanced_class_weights
from src.utils.config import get_dir, load_config
from src.utils.logger import get_logger
from src.utils.runtime import get_device, set_seed
from src.utils.tracking import tracked_run

log = get_logger(__name__)


def build_model(model_type: str, vocab_size: int, neural_cfg: dict) -> tuple[nn.Module, dict]:
    """Create the model and return it with the kwargs needed to rebuild it at inference."""
    common = {
        "vocab_size": vocab_size,
        "embed_dim": neural_cfg["embed_dim"],
        "num_classes": NUM_CLASSES,
        "dropout": neural_cfg["dropout"],
        "pad_idx": PAD_ID,
    }
    if model_type == "cnn":
        kwargs = {**common, "num_filters": neural_cfg["cnn"]["num_filters"],
                  "kernel_sizes": list(neural_cfg["cnn"]["kernel_sizes"])}
        return TextCNN(**kwargs), kwargs
    if model_type == "bilstm":
        kwargs = {**common, "hidden_dim": neural_cfg["bilstm"]["hidden_dim"],
                  "num_layers": neural_cfg["bilstm"]["num_layers"]}
        return BiLSTMClassifier(**kwargs), kwargs
    raise ValueError(f"Unknown model '{model_type}'. Choose from {NEURAL_MODELS}")


def make_loader(frame, vocab: Vocabulary, max_len: int, batch_size: int, shuffle: bool) -> DataLoader:
    ids = torch.tensor([vocab.encode(t, max_len) for t in frame["text_clean"]], dtype=torch.long)
    labels = torch.tensor(frame["label"].to_numpy(), dtype=torch.long)
    # num_workers=0: worker processes add nothing for in-memory tensors and avoid
    # Windows multiprocessing ("spawn") pitfalls.
    return DataLoader(TensorDataset(ids, labels), batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (mean loss, true labels, class probabilities)."""
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits = model(input_ids)
        total_loss += loss_fn(logits, labels).item() * labels.size(0)
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    y_true = np.concatenate(all_labels)
    return total_loss / len(y_true), y_true, np.concatenate(all_probs)


def train(model_type: str, cfg: dict, epochs: int | None = None) -> dict:
    neural_cfg = cfg["neural"]
    epochs = epochs or neural_cfg["epochs"]
    set_seed(cfg["project"]["seed"])
    device = get_device()
    log.info("Training %s on %s", model_type, device)

    splits = load_splits(cfg)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    vocab = Vocabulary.build(train_df["text_clean"], neural_cfg["min_freq"], neural_cfg["max_vocab"])
    log.info("Vocabulary size: %d", len(vocab))

    max_len, batch_size = neural_cfg["max_len"], neural_cfg["batch_size"]
    train_loader = make_loader(train_df, vocab, max_len, batch_size, shuffle=True)
    val_loader = make_loader(val_df, vocab, max_len, batch_size, shuffle=False)
    test_loader = make_loader(test_df, vocab, max_len, batch_size, shuffle=False)

    model, model_kwargs = build_model(model_type, len(vocab), neural_cfg)
    model.to(device)

    weights = None
    if neural_cfg.get("use_class_weights", True):
        weights = torch.tensor(balanced_class_weights(train_df["label"]), dtype=torch.float32, device=device)
        log.info("Class weights: %s", weights.tolist())
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=neural_cfg["lr"])

    params = {**{k: v for k, v in neural_cfg.items() if not isinstance(v, dict)}, **model_kwargs, "epochs": epochs}
    history: list[dict] = []
    best_f1, best_state, best_epoch, stale = -1.0, None, 0, 0

    with tracked_run(cfg, model_type, params) as tracker:
        for epoch in range(1, epochs + 1):
            model.train()
            running, seen = 0.0, 0
            for input_ids, labels in train_loader:
                input_ids, labels = input_ids.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(input_ids), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # LSTMs can have exploding gradients
                optimizer.step()
                running += loss.item() * labels.size(0)
                seen += labels.size(0)

            val_loss, y_val, val_probs = evaluate(model, val_loader, loss_fn, device)
            val_metrics = compute_metrics(y_val, val_probs.argmax(axis=1))
            record = {
                "epoch": epoch,
                "train_loss": running / seen,
                "val_loss": val_loss,
                "val_f1_macro": val_metrics["f1_macro"],
                "val_accuracy": val_metrics["accuracy"],
            }
            history.append(record)
            tracker.log_metrics({k: v for k, v in record.items() if k != "epoch"}, step=epoch)
            log.info("epoch %d | train loss %.4f | val loss %.4f | val macro-F1 %.4f",
                     epoch, record["train_loss"], val_loss, val_metrics["f1_macro"])

            if val_metrics["f1_macro"] > best_f1:
                best_f1, best_epoch, stale = val_metrics["f1_macro"], epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= neural_cfg["patience"]:
                    log.info("Early stopping: no validation improvement for %d epochs.", stale)
                    break

        # Restore the best epoch, then touch the test set exactly once.
        model.load_state_dict(best_state)
        _, y_val, val_probs = evaluate(model, val_loader, loss_fn, device)
        _, y_test, test_probs = evaluate(model, test_loader, loss_fn, device)
        val_metrics = compute_metrics(y_val, val_probs.argmax(axis=1))
        test_metrics = compute_metrics(y_test, test_probs.argmax(axis=1))
        tracker.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    checkpoint = {
        "model_type": model_type,
        "model_kwargs": model_kwargs,
        "state_dict": {k: v.cpu() for k, v in best_state.items()},
        "vocab": vocab.to_dict(),
        "max_len": max_len,
    }
    model_path = get_dir(cfg, "models_dir") / f"{model_type}.pt"
    torch.save(checkpoint, model_path)

    figures = get_dir(cfg, "figures_dir")
    plot_history(history, model_type, figures / f"history_{model_type}.png")
    plot_confusion_matrix(test_metrics["confusion_matrix"], model_type, figures / f"confusion_{model_type}.png")
    save_results(cfg, model_type, val_metrics, test_metrics, params,
                 {"history": history, "best_epoch": best_epoch, "device": str(device)})
    save_predictions(cfg, model_type, test_df["text"], y_test, test_probs)
    with open(get_dir(cfg, "metrics_dir") / f"{model_type}_history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    log.info("%s | best epoch %d | TEST macro-F1 %.4f, accuracy %.4f | saved %s",
             model_type, best_epoch, test_metrics["f1_macro"], test_metrics["accuracy"], model_path)
    return test_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the CNN or BiLSTM text classifier")
    parser.add_argument("--model", required=True, choices=NEURAL_MODELS)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    try:
        train(args.model, load_config(args.config), args.epochs)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
