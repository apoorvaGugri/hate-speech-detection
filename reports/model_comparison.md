# Model comparison (test set)

Sorted by macro-F1, the primary metric.

| model | f1_macro | f1_weighted | accuracy | precision_macro | recall_macro | f1_hate_speech | recall_hate_speech | f1_offensive | f1_neither | val_f1_macro | smoke_test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bert | 0.7965 | 0.9165 | 0.9105 | 0.7686 | 0.8394 | 0.5330 | 0.6643 | 0.9463 | 0.9102 | 0.7795 | False |
| logistic_regression | 0.7462 | 0.8819 | 0.8727 | 0.7055 | 0.8128 | 0.4738 | 0.6143 | 0.9203 | 0.8444 | 0.7455 | False |
| cnn | 0.7270 | 0.8778 | 0.8666 | 0.6923 | 0.7868 | 0.4127 | 0.5571 | 0.9184 | 0.8498 | 0.7394 | False |
| bilstm | 0.7223 | 0.8755 | 0.8580 | 0.6910 | 0.7981 | 0.4000 | 0.6286 | 0.9162 | 0.8509 | 0.7197 | False |
| linear_svm | 0.7126 | 0.8857 | 0.8937 | 0.7684 | 0.6895 | 0.3732 | 0.2786 | 0.9361 | 0.8285 | 0.6848 | False |
| naive_bayes | 0.5781 | 0.8389 | 0.8658 | 0.7828 | 0.5466 | 0.1046 | 0.0571 | 0.9218 | 0.7081 | 0.5713 | False |
