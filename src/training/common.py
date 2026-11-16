"""Helpers shared by the PyTorch training scripts."""
from __future__ import annotations

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.constants import LABEL_IDS


def balanced_class_weights(labels) -> np.ndarray:
    """Weights n_samples / (n_classes * n_class_samples) for the loss function.

    Passed to ``CrossEntropyLoss(weight=...)`` so that errors on rare classes
    cost more. Computed from the TRAINING labels only.
    """
    labels = np.asarray(labels)
    present = np.unique(labels)
    weights = np.ones(len(LABEL_IDS), dtype=np.float32)
    computed = compute_class_weight("balanced", classes=present, y=labels)
    for cls, weight in zip(present, computed):
        weights[int(cls)] = weight
    return weights
