"""All figures. Uses the non-interactive Agg backend so scripts run headless."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from src.constants import ID2LABEL, LABEL_NAMES  # noqa: E402

CLASS_COLORS = ["#B91C1C", "#D97706", "#047857"]   # hate, offensive, neither
sns.set_theme(style="whitegrid", font_scale=1.0)


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_distribution(class_counts: dict[str, int], path) -> Path:
    names, values = list(class_counts), list(class_counts.values())
    total = max(sum(values), 1)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(names, values, color=CLASS_COLORS[: len(names)])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:,}\n({100 * value / total:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Tweets")
    ax.set_title("Class distribution")
    ax.set_ylim(0, max(values) * 1.18)
    return _save(fig, path)


def plot_length_distribution(word_counts, labels, path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    word_counts, labels = np.asarray(word_counts), np.asarray(labels)
    bins = np.arange(0, max(int(np.percentile(word_counts, 99)), 10) + 2)
    for idx, name in ID2LABEL.items():
        subset = word_counts[labels == idx]
        if len(subset):
            ax.hist(subset, bins=bins, density=True, histtype="step", linewidth=1.8,
                    color=CLASS_COLORS[idx], label=name)
    ax.set_xlabel("Words per tweet")
    ax.set_ylabel("Density (per class)")
    ax.set_title("Tweet length by class")
    ax.legend()
    return _save(fig, path)


def plot_confusion_matrix(cm, title: str, path) -> Path:
    """Counts (left) and row-normalised recall view (right)."""
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    normalised = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    axes[0].set_title(f"{title} - counts")
    sns.heatmap(normalised, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=axes[1],
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    axes[1].set_title(f"{title} - share of true class (recall on diagonal)")
    for ax in axes:
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=15)
        ax.tick_params(axis="y", rotation=0)
    return _save(fig, path)


def plot_history(history: list[dict], title: str, path) -> Path:
    """Training vs validation loss per epoch, plus validation macro-F1."""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], marker="o", label="Training loss", color="#4338CA")
    axes[0].plot(epochs, [h["val_loss"] for h in history], marker="o", label="Validation loss", color="#D97706")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title(f"{title} - loss")
    axes[0].legend()
    axes[1].plot(epochs, [h["val_f1_macro"] for h in history], marker="o", color="#047857")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation macro-F1")
    axes[1].set_title(f"{title} - validation macro-F1")
    for ax in axes:
        ax.set_xticks(epochs)
    return _save(fig, path)


def plot_model_comparison(table, path) -> Path:
    """Grouped bars: macro-F1, hate-speech F1 and accuracy for each model (test set)."""
    metrics = [("f1_macro", "Macro-F1"), ("f1_hate_speech", "Hate-speech F1"), ("accuracy", "Accuracy")]
    models = list(table["model"])
    x = np.arange(len(models))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(models)), 4.5))
    for k, (column, label) in enumerate(metrics):
        values = table[column].to_numpy(dtype=float)
        bars = ax.bar(x + (k - 1) * width, values, width, label=label, color=["#4338CA", "#B91C1C", "#9CA3AF"][k])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylim(0, 1.22)                       # head-room so the legend never covers a bar
    ax.set_ylabel("Score (test set)")
    ax.set_title("Model comparison")
    ax.legend(loc="upper center", ncol=len(metrics), frameon=False)
    return _save(fig, path)


def plot_robustness(rows: list[dict], title: str, path) -> Path:
    names = [r["perturbation"] for r in rows]
    values = [r["f1_macro"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(names, values, color=["#4338CA"] + ["#9CA3AF"] * (len(rows) - 1))
    for bar, value in zip(bars, values):
        ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.set_xlabel("Macro-F1")
    ax.set_title(title)
    return _save(fig, path)
