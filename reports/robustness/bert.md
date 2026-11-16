# Robustness - bert

Stratified test sample: 500 tweets.

| Perturbation | Macro-F1 | Hate-speech recall | Predictions flipped |
|---|---:|---:|---:|
| clean | 0.7716 | 0.5172 | 0.0000 |
| typo_swap | 0.6940 | 0.4828 | 0.0880 |
| leetspeak | 0.6860 | 0.4828 | 0.0940 |
| char_separator | 0.6984 | 0.4828 | 0.0840 |
| uppercase | 0.7716 | 0.5172 | 0.0000 |
| benign_suffix | 0.7823 | 0.5172 | 0.0120 |
