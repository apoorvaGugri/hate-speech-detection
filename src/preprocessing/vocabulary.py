"""Word-level vocabulary for the CNN and BiLSTM.

The vocabulary is built from the TRAINING split only. Building it from all data
would leak information about which words occur in validation/test.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1


class Vocabulary:
    def __init__(self, token_to_id: dict[str, int] | None = None):
        self.token_to_id = token_to_id or {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 2, max_size: int = 30000) -> "Vocabulary":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(text.split())
        vocab = cls()
        # most_common is deterministic for equal counts (insertion order), so the
        # same data + seed always yields the same ids.
        for token, freq in counts.most_common():
            if freq < min_freq or len(vocab) >= max_size:
                break
            vocab.token_to_id[token] = len(vocab.token_to_id)
        return vocab

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, max_len: int) -> list[int]:
        """Text -> fixed-length list of ids (truncated or right-padded)."""
        ids = [self.token_to_id.get(tok, UNK_ID) for tok in text.split()[:max_len]]
        return ids + [PAD_ID] * (max_len - len(ids))

    def to_dict(self) -> dict[str, int]:
        return dict(self.token_to_id)

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "Vocabulary":
        return cls({str(k): int(v) for k, v in data.items()})
