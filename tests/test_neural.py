"""PyTorch tests. Skipped automatically when torch is not installed."""
import pytest

torch = pytest.importorskip("torch")

from src.inference.predictor import Predictor  # noqa: E402
from src.models.bilstm import BiLSTMClassifier  # noqa: E402
from src.models.cnn import TextCNN  # noqa: E402
from src.training.train_neural import train  # noqa: E402


@pytest.mark.parametrize("model_cls", [TextCNN, BiLSTMClassifier])
def test_forward_shape(model_cls):
    model = model_cls(vocab_size=50, embed_dim=16)
    ids = torch.randint(2, 50, (4, 12))
    ids[0, 6:] = 0          # padded row
    ids[1, :] = 0           # all-padding row must not crash
    assert model(ids).shape == (4, 3)


def test_bilstm_ignores_padding():
    model = BiLSTMClassifier(vocab_size=50, embed_dim=16, dropout=0.0).eval()
    short = torch.tensor([[5, 6, 7]])
    padded = torch.tensor([[5, 6, 7, 0, 0, 0]])
    assert torch.allclose(model(short), model(padded), atol=1e-6)


@pytest.mark.parametrize("model_type", ["cnn", "bilstm"])
def test_train_save_and_reload(prepared_cfg, model_type):
    metrics = train(model_type, prepared_cfg, epochs=2)
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    predictor = Predictor(model_type, prepared_cfg)
    probs = predictor.predict_proba(["cedar coral cloud", "amber arrow anchor"])
    assert probs.shape == (2, 3)
    assert probs.sum(axis=1) == pytest.approx([1.0, 1.0], abs=1e-5)
