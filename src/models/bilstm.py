"""Bidirectional LSTM classifier.

An LSTM reads the tweet token by token and carries a memory cell guarded by
gates, which lets it keep information across longer spans than a CNN window.
"Bidirectional" runs one LSTM left-to-right and another right-to-left; we
concatenate their final hidden states so the representation of the tweet has
seen context from both sides.

``pack_padded_sequence`` tells the LSTM the true length of every tweet so that
padding tokens are skipped - otherwise the final hidden state of the forward
LSTM would be computed after reading a run of <pad> tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 3,
        num_layers: int = 1,
        dropout: float = 0.5,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,   # inter-layer dropout needs >1 layer
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # True lengths; clamp so an all-padding row does not crash packing.
        lengths = (input_ids != self.pad_idx).sum(dim=1).clamp(min=1).cpu()
        embedded = self.dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        # hidden: (num_layers * 2, batch, hidden_dim). The last two entries are the
        # top layer's forward and backward final states.
        features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(features))
