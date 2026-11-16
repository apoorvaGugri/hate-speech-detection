"""Project-wide constants: the label scheme and the project root."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Label ids exactly as published in the Davidson et al. (2017) dataset.
ID2LABEL = {0: "Hate speech", 1: "Offensive language", 2: "Neither"}
LABEL2ID = {name: idx for idx, name in ID2LABEL.items()}
LABEL_IDS = sorted(ID2LABEL)
LABEL_NAMES = [ID2LABEL[i] for i in LABEL_IDS]
NUM_CLASSES = len(LABEL_IDS)

BASELINE_MODELS = ["logistic_regression", "linear_svm", "naive_bayes"]
NEURAL_MODELS = ["cnn", "bilstm"]
