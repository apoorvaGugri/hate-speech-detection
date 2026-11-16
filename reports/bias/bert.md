# Bias audit - bert

Rates are shares of the stated n. Small n = noisy.

## Test-set false positives by identity mention

| Group | Tweets | Neither flagged (n) | Non-hate called hate (n) | Hate missed (n) |
|---|---:|---:|---:|---:|
| any_identity_term | 242 | 0.1538 (39) | 0.107 (215) | 0.1111 (27) |
| gender | 157 | 0.2 (25) | 0.0654 (153) | 0.5 (4) |
| sexual_orientation_gender_identity | 21 | 0.5 (2) | 0.3333 (15) | 0.0 (6) |
| religion | 10 | 0.0 (3) | 0.1667 (6) | 0.0 (4) |
| race_ethnicity_origin | 63 | 0.0833 (12) | 0.2083 (48) | 0.0667 (15) |
| disability | 2 | None (0) | 0.0 (2) | None (0) |
| no_identity_term | 2194 | 0.0622 (370) | 0.0447 (2081) | 0.3894 (113) |

## Template probe (all sentences are harmless; expected label: Neither)

- Identity sentences flagged: **0.359**
- Control sentences flagged: **0.0** (n=5)

| Term | Sentences | Flagged | Flagged as hate |
|---|---:|---:|---:|
| gay | 6 | 6 | 6 |
| lesbian | 6 | 6 | 6 |
| transgender | 6 | 6 | 6 |
| muslim | 6 | 1 | 1 |
| jewish | 6 | 1 | 1 |
| christian | 6 | 0 | 0 |
| hindu | 6 | 0 | 0 |
| black | 6 | 2 | 2 |
| white | 6 | 2 | 2 |
| asian | 6 | 1 | 1 |
| latino | 6 | 2 | 2 |
| disabled | 6 | 0 | 0 |
| deaf | 6 | 1 | 1 |

Label ids: {0: 'Hate speech', 1: 'Offensive language', 2: 'Neither'}
