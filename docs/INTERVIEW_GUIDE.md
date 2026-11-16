# Interview guide

Short answers to the questions this project invites. Each points to the file where the decision lives.

## The problem

**Why three classes instead of hate / not hate?**
The dataset's main finding is that offensive language is not the same as hate speech. Swearing is
common and mostly harmless; hate speech targets a group. A binary model trained on "contains bad
words" flags both. Keeping "offensive" as its own class forces the model to learn the difference
and lets a platform treat the two differently.

**What makes it hard?**
Class imbalance (hate is about 6%), subjective labels, very short texts, and meaning that depends
on speaker and context (reclaimed terms, quotes, sarcasm).

## Data and leakage (`src/data/split.py`)

**How did you prevent data leakage?**
1. Duplicates are removed on *cleaned* text before splitting, so the same retweet cannot sit in
   train and test.
2. The split is stratified and saved to disk; every model uses the same rows.
3. Anything that is *fitted* (TF-IDF vocabulary and IDF, the word vocabulary, class weights, model
   weights) sees the training split only. The TF-IDF vectoriser lives inside the sklearn `Pipeline`
   for exactly this reason.
4. Hyper-parameters and early stopping use the validation split. The test split is evaluated once
   per model at the end.

**Why stratified?** With a 6% class, a random 10% test split could hold noticeably more or fewer
hate tweets than the training data, making test metrics unstable.

## Preprocessing (`src/preprocessing/text_cleaner.py`)

**Why two cleaning functions?**
BERT was pre-trained on natural sentences and has its own WordPiece tokenizer, so it gets light
cleaning: decode HTML, drop `RT`, mask handles as `@user` and URLs as `http`. TF-IDF, CNN and
BiLSTM build a vocabulary from our data, so lower-casing, removing punctuation and squashing
"soooo" shrink the vocabulary and reduce sparsity.

**Why keep stop words?** Words like "you", "they", "all", "not" tell you whether someone is being
targeted. Removing them deletes the signal that separates hate from offensive.

**Why mask handles?** They are identifiers, not content: they cause overfitting to specific
accounts and are personal data.

## Imbalance

**How did you handle it?** Class-weighted loss (`class_weight="balanced"` in sklearn, `weight=` in
`CrossEntropyLoss`): weight = n_samples / (n_classes × n_class_samples). Errors on hate tweets cost
roughly 13 times more than errors on offensive tweets. Selection and early stopping use macro-F1.

**Why not SMOTE or oversampling?** SMOTE interpolates between feature vectors, which is not
meaningful for text. Random oversampling duplicates the same few tweets and encourages
memorisation. Weighted loss has the same effect on the gradient without changing the data.

**Trade-off to mention:** weighting raises hate-speech recall and lowers its precision. Which
matters more is a policy decision, which is why per-class precision and recall are both reported.

## Metrics (`src/evaluation/metrics.py`)

- **Accuracy**: share correct. Misleading here; "always offensive" scores about 77%.
- **Precision** (per class): of the tweets flagged as hate, how many were hate. Low precision =
  wrongful flags.
- **Recall** (per class): of the true hate tweets, how many were caught. Low recall = hate gets through.
- **F1**: harmonic mean of the two; punishes a model that is good at only one.
- **Macro-F1**: plain average of the three per-class F1 scores. Every class counts equally. Primary metric.
- **Weighted-F1**: average weighted by class size. Dominated by the offensive class; behaves like accuracy.
- **Confusion matrix**: rows are true classes, columns predictions. The cell to look at first is
  true hate → predicted offensive.

## The models

**TF-IDF + linear models (`src/models/baselines.py`).** Each tweet becomes a sparse vector of
uni- and bi-gram weights: term frequency × inverse document frequency, so words frequent in one
tweet but rare overall score highest. Logistic regression and linear SVM learn one weight per
n-gram per class. Fast, strong, and fully interpretable through coefficients. Blind to word order
beyond bigrams and to any word not seen in training. Naive Bayes assumes words are independent
given the class and has no class weighting, so it tends to under-predict the rare class.

**TextCNN (`src/models/cnn.py`).** Embedding → parallel 1-D convolutions with kernel sizes 3, 4, 5
→ max-pool over time → linear layer. Each filter is a learned n-gram detector; max-pooling asks
"did this pattern occur anywhere". Good at key phrases, no long-range structure.

**BiLSTM (`src/models/bilstm.py`).** Embedding → LSTM read in both directions → concatenated final
hidden states → linear layer. Gates let it carry information across the sentence.
`pack_padded_sequence` makes it skip padding. Sequential, so slower than a CNN, and with ~20k
training tweets and embeddings learned from scratch it has little to learn from.

**BERT (`src/models/bert_classifier.py`, `src/training/train_bert.py`).**
- *Architecture*: 12 Transformer encoder layers, hidden size 768, 12 heads, ~110M parameters.
  Input is `[CLS] wordpieces [SEP]`; each token embedding = token + position + segment.
- *Self-attention*: every token computes a weighted mix of all tokens in the sentence, in both
  directions, in every layer. The vector for a word therefore depends on its context. That is
  the difference from static embeddings, where a word has one vector everywhere.
- *Pre-training*: masked-language modelling on large unlabeled corpora teaches general English.
- *Fine-tuning*: a new linear layer (768 → 3) is put on the pooled `[CLS]` vector and **all**
  weights are updated on our labels for 3 epochs.
- *Why lr 2e-5 and warm-up?* The pre-trained weights are already good; a large learning rate
  destroys them (catastrophic forgetting). Warm-up keeps early updates small while the randomly
  initialised head is still producing noisy gradients.
- *Why AdamW with weight decay excluded for bias/LayerNorm?* Standard practice from the BERT
  paper: decay regularises weight matrices; applying it to biases and normalisation scales hurts.
- *Why gradient clipping?* Protects against an occasional huge gradient step.
- *Why dynamic padding?* Tweets are short; padding each batch to its own longest sequence is
  several times faster than padding everything to `max_len`.
- *Why mixed precision only on GPU?* float16 autocast is a CUDA speed-up; on CPU it gains nothing.
- *Why uncased?* Tweet capitalisation is erratic; the uncased model handles "HATE" and "hate" alike.

**Why a custom loop instead of `Trainer`?** To use a class-weighted loss directly and to be able
to explain each line: forward, loss, backward, unscale, clip, optimizer step, scheduler step.

**What did the comparison show?** (numbers from `reports/model_comparison.md`)
BERT won: macro-F1 0.797 against 0.746 for logistic regression, 0.727 for the CNN and 0.722 for
the BiLSTM. Two things worth saying out loud: (1) the linear TF-IDF model beat both neural
networks trained from scratch, because on lexicon-sampled data single words carry most of the
signal and 19k tweets is little data for learning embeddings; (2) BERT's gain is concentrated
where context matters, with hate-speech F1 rising from 0.47 to 0.53. Also be honest about what
did not work: hate-speech precision is only 0.44, and the bias audit shows BERT labels harmless
sentences containing "gay", "lesbian" or "transgender" as hate speech every time.

**Is a small gain worth a 110M-parameter model?** A fair challenge. Answer with latency, cost,
the error analysis (which errors BERT fixes) and the robustness report, not with the headline score.

## Explainability (`src/explainability/lime_explainer.py`)

**How does LIME work?** It makes ~1000 copies of the text with random words removed, gets the
model's probabilities for each, and fits a small weighted linear model to them. Its coefficients
are the word weights.

**Why LIME over SHAP?** One `predict_proba` function makes it work identically for all six
models, and it is cheap enough for a CPU demo. SHAP has stronger theoretical guarantees but its
transformer explainer is much slower.

**Limits you should volunteer:** it is a local approximation; results change between runs; it
assumes words contribute independently, which is false for BERT; it shows sensitivity to
deletion, not reasoning. For logistic regression, the coefficients are an exact global
explanation, and comparing them with LIME is a useful sanity check.

## Bias and robustness (`src/evaluation/bias_audit.py`, `robustness.py`)

**What bias did you test for?** Whether harmless text that mentions an identity group is flagged
more often: false-positive rates on the test set with and without identity terms, plus template
sentences such as "I am <identity>." that should always be "neither".

**What could you not test?** Dialect bias. Prior work shows models trained on this dataset flag
African-American English more often, but the data has no dialect labels, so this project can
only cite that risk, not measure it.

**Robustness?** Typos, leetspeak, dotted letters, upper-casing and appended friendly text, each
scored by macro-F1 and by how many predictions flip. Sub-word models usually degrade more
gracefully than word-level ones, because an unseen misspelling becomes `<unk>` for TF-IDF/CNN/LSTM
but is split into known pieces by WordPiece. Measured for BERT only: typos, leetspeak and dotted
letters each cost about 0.08 macro-F1 and changed 8-9% of predictions; upper-casing changed none.
The other models were not tested, so the comparison between families is still a hypothesis.

## Engineering

- **Config-driven** (`configs/config.yaml`): an experiment is a commit plus a config.
- **One inference path** (`src/inference/predictor.py`): the app, LIME, the bias audit and the
  robustness test all call the same `Predictor`, which applies the same cleaning as training.
  This prevents train/serve skew.
- **Tracking**: MLflow logs parameters and metrics per run; it is wrapped so that a tracking
  failure never stops training.
- **Tests**: unit tests for cleaning, splitting, metrics; an end-to-end test that trains, saves,
  reloads and serves the baselines on a synthetic corpus.
- **What you would do next**: confidence calibration and a "send to human" threshold;
  cross-dataset evaluation (HateCheck functional tests); a domain model such as BERTweet;
  5-fold cross-validation to put confidence intervals on the hate-speech F1.
