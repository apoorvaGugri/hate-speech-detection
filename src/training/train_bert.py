"""Phase 5 - fine-tune BERT (the main model).

    python -m src.training.train_bert                      # full run (GPU strongly recommended)
    python -m src.training.train_bert --max-train-samples 2000 --epochs 1 --output-name bert_smoke
    python -m src.training.train_bert --model-name distilbert-base-uncased --output-name distilbert

A hand-written PyTorch loop is used instead of ``transformers.Trainer`` so every
step is visible and explainable:

  tokenise -> forward -> class-weighted cross-entropy -> backward ->
  clip gradients -> AdamW step -> linear warm-up/decay schedule step

Fine-tuning choices
  * lr 2e-5, 3 epochs, batch 16 - the range recommended in the BERT paper.
  * AdamW with weight decay 0.01, not applied to biases and LayerNorm weights.
  * Warm-up for the first 10% of steps: the new classifier head starts random,
    and large early updates would damage the pre-trained layers.
  * Dynamic padding: each batch is padded to its own longest tweet, not to max_len.
  * Mixed precision (AMP) on CUDA only: ~2x faster and less GPU memory.
  * Best checkpoint chosen by VALIDATION macro-F1; the test set is used once at the end.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.split import load_splits
from src.evaluation.metrics import compute_metrics, save_predictions, save_results
from src.evaluation.plots import plot_confusion_matrix, plot_history
from src.models.bert_classifier import build_bert
from src.training.common import balanced_class_weights
from src.utils.config import get_dir, load_config
from src.utils.logger import get_logger
from src.utils.runtime import get_device, set_seed
from src.utils.tracking import tracked_run

log = get_logger(__name__)


class TweetDataset(Dataset):
    """Holds raw strings; tokenisation happens per batch in the collate function."""

    def __init__(self, texts, labels):
        self.texts = [str(t) for t in texts]
        self.labels = [int(l) for l in labels]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        return self.texts[idx], self.labels[idx]


def make_collate(tokenizer, max_len: int):
    def collate(batch):
        texts, labels = zip(*batch)
        encoded = tokenizer(list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        return dict(encoded), torch.tensor(labels, dtype=torch.long)

    return collate


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)       # torch >= 2.3
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)          # older torch


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_amp: bool) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    for inputs, labels in tqdm(loader, desc="eval", leave=False):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(**inputs).logits
        logits = logits.float()
        total_loss += loss_fn(logits, labels).item() * labels.size(0)
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    y_true = np.concatenate(all_labels)
    return total_loss / len(y_true), y_true, np.concatenate(all_probs)


def _subsample(frame, n: int, seed: int):
    if n >= len(frame):
        return frame
    subset, _ = train_test_split(frame, train_size=n, stratify=frame["label"], random_state=seed)
    return subset.reset_index(drop=True)


def train(cfg: dict, args: argparse.Namespace) -> dict:
    bert_cfg = dict(cfg["bert"])
    if args.model_name:
        bert_cfg["model_name"] = args.model_name
    if args.epochs:
        bert_cfg["epochs"] = args.epochs
    if args.batch_size:
        bert_cfg["batch_size"] = args.batch_size
    seed = cfg["project"]["seed"]
    set_seed(seed)
    device = get_device()
    use_amp = device.type == "cuda"
    smoke_test = args.max_train_samples is not None

    splits = load_splits(cfg)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    if smoke_test:
        train_df = _subsample(train_df, args.max_train_samples, seed)
        log.warning("SMOKE TEST: training on %d tweets only. Do not report these numbers as results.", len(train_df))
    elif device.type == "cpu":
        log.warning(
            "No CUDA GPU detected. Full BERT fine-tuning on a CPU can take several hours. "
            "Options: Google Colab (notebooks/colab_train_bert.ipynb), --max-train-samples N for a "
            "smoke test, or --model-name distilbert-base-uncased."
        )

    log.info("Loading %s (first run downloads the weights)...", bert_cfg["model_name"])
    tokenizer, model = build_bert(bert_cfg["model_name"])
    model.to(device)

    collate = make_collate(tokenizer, bert_cfg["max_len"])
    train_loader = DataLoader(TweetDataset(train_df["text_light"], train_df["label"]),
                              batch_size=bert_cfg["batch_size"], shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(TweetDataset(val_df["text_light"], val_df["label"]),
                            batch_size=bert_cfg["eval_batch_size"], shuffle=False, collate_fn=collate, num_workers=0)
    test_loader = DataLoader(TweetDataset(test_df["text_light"], test_df["label"]),
                             batch_size=bert_cfg["eval_batch_size"], shuffle=False, collate_fn=collate, num_workers=0)

    weights = None
    if bert_cfg.get("use_class_weights", True):
        weights = torch.tensor(balanced_class_weights(train_df["label"]), dtype=torch.float32, device=device)
        log.info("Class weights: %s", weights.tolist())
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    no_decay = ("bias", "LayerNorm.weight")
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": bert_cfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=bert_cfg["lr"])

    from transformers import get_linear_schedule_with_warmup

    total_steps = len(train_loader) * bert_cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(bert_cfg["warmup_ratio"] * total_steps), num_training_steps=total_steps
    )
    scaler = _make_grad_scaler(use_amp)

    output_dir = get_dir(cfg, "models_dir") / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    params = {**bert_cfg, "device": str(device), "train_rows": len(train_df), "smoke_test": smoke_test}
    history: list[dict] = []
    best_f1, best_epoch, stale = -1.0, 0, 0

    with tracked_run(cfg, args.output_name, params) as tracker:
        for epoch in range(1, bert_cfg["epochs"] + 1):
            model.train()
            running, seen = 0.0, 0
            progress = tqdm(train_loader, desc=f"epoch {epoch}/{bert_cfg['epochs']}")
            for inputs, labels in progress:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(**inputs).logits
                loss = loss_fn(logits.float(), labels)     # loss in float32 for numerical stability
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)                 # so clipping sees true gradient norms
                nn.utils.clip_grad_norm_(model.parameters(), bert_cfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                running += loss.item() * labels.size(0)
                seen += labels.size(0)
                progress.set_postfix(loss=f"{running / seen:.4f}")

            val_loss, y_val, val_probs = evaluate(model, val_loader, loss_fn, device, use_amp)
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
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                log.info("New best model saved to %s", output_dir)
            else:
                stale += 1
                if stale >= bert_cfg["patience"]:
                    log.info("Early stopping: no validation improvement for %d epochs.", stale)
                    break

        # Reload the best checkpoint and evaluate on the held-out test set once.
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
        _, y_val, val_probs = evaluate(model, val_loader, loss_fn, device, use_amp)
        _, y_test, test_probs = evaluate(model, test_loader, loss_fn, device, use_amp)
        val_metrics = compute_metrics(y_val, val_probs.argmax(axis=1))
        test_metrics = compute_metrics(y_test, test_probs.argmax(axis=1))
        tracker.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    with open(output_dir / "training_meta.json", "w", encoding="utf-8") as fh:
        json.dump({"base_model": bert_cfg["model_name"], "max_len": bert_cfg["max_len"],
                   "best_epoch": best_epoch, "smoke_test": smoke_test}, fh, indent=2)

    figures = get_dir(cfg, "figures_dir")
    plot_history(history, args.output_name, figures / f"history_{args.output_name}.png")
    plot_confusion_matrix(test_metrics["confusion_matrix"], args.output_name,
                          figures / f"confusion_{args.output_name}.png")
    save_results(cfg, args.output_name, val_metrics, test_metrics, params,
                 {"history": history, "best_epoch": best_epoch, "smoke_test": smoke_test})
    save_predictions(cfg, args.output_name, test_df["text"], y_test, test_probs)

    log.info("%s | best epoch %d | TEST macro-F1 %.4f, accuracy %.4f",
             args.output_name, best_epoch, test_metrics["f1_macro"], test_metrics["accuracy"])
    return test_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune BERT for 3-class hate speech detection")
    parser.add_argument("--model-name", default=None, help="Hugging Face model id (default from config)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="train on a stratified subset (smoke test on CPU)")
    parser.add_argument("--output-name", default="bert", help="folder under models/ and name in reports")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    try:
        train(load_config(args.config), args)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    except OSError as exc:
        log.error("Could not load or save the model (%s). The first run needs internet access "
                  "to download the pre-trained weights from huggingface.co.", exc)
        return 1
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            log.error("Out of memory. Re-run with a smaller batch, e.g. --batch-size 8.")
            return 1
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
