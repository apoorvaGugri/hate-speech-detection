"""BERT sequence classifier (the main model).

Architecture in one paragraph
-----------------------------
``bert-base-uncased`` is a stack of 12 Transformer *encoder* layers (hidden size
768, 12 attention heads, ~110M parameters). Text is split into WordPiece
sub-words, wrapped as ``[CLS] tokens [SEP]``, and each token's embedding is the
sum of token + position + segment embeddings. In every layer, self-attention
lets each token look at every other token in both directions, so the vector for
a word depends on its whole context - this is what "context-aware" means and
what TF-IDF, CNN and (largely) LSTM lack.

Pre-training vs fine-tuning
---------------------------
BERT was pre-trained on unlabeled text with masked-language modelling. For
classification, ``AutoModelForSequenceClassification`` adds one randomly
initialised linear layer (768 -> 3) on top of the pooled ``[CLS]`` vector.
Fine-tuning updates ALL weights end-to-end with a small learning rate (2e-5):
large enough to adapt, small enough not to erase what pre-training learned.
"""
from __future__ import annotations

from src.constants import ID2LABEL, LABEL2ID, NUM_CLASSES


def build_bert(model_name: str = "bert-base-uncased"):
    """Return (tokenizer, model). The first call downloads ~440 MB to the HF cache."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return tokenizer, model
