"""Reproducibility and hardware detection.

Run ``python -m src.utils.runtime`` to see what this machine can do.
torch is imported lazily so the classical baselines work without it.
"""
from __future__ import annotations

import os
import platform
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    """Return the best available torch device: CUDA GPU if present, else CPU."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_runtime() -> dict:
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            )
    except ImportError:
        info["torch"] = "not installed"
        info["cuda_available"] = False
    return info


if __name__ == "__main__":
    details = describe_runtime()
    for key, value in details.items():
        print(f"{key:>16}: {value}")
    if details["torch"] == "not installed":
        print("\nPyTorch is not installed. See README -> 'Install PyTorch'.")
    elif details["cuda_available"]:
        print("\nNVIDIA GPU detected: BERT fine-tuning can run locally.")
    else:
        print(
            "\nNo CUDA GPU detected. Baselines, CNN and BiLSTM train fine on CPU.\n"
            "Full BERT fine-tuning on CPU takes hours; use Google Colab "
            "(notebooks/colab_train_bert.ipynb) or a smoke test with --max-train-samples."
        )
