"""TextCNN (Kim, 2014).

Idea: a 1-D convolution with kernel size k slides over the word embeddings and
acts as a learned k-gram detector. Max-pooling over time keeps the strongest
activation of each filter, i.e. "did this pattern occur anywhere in the tweet?".
Using kernel sizes 3, 4 and 5 in parallel captures phrases of different lengths.

Strength: fast, good at local key phrases. Weakness: no notion of long-range
order or context - "I was called X" and "you are X" look similar to it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_classes: int = 3,
        num_filters: int = 100,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.5,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2) for k in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch, seq_len)
        x = self.embedding(input_ids)            # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)                    # Conv1d wants (batch, channels, seq_len)
        pooled = [F.relu(conv(x)).max(dim=2).values for conv in self.convs]   # each (batch, num_filters)
        features = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(features)                 # raw logits, (batch, num_classes)
