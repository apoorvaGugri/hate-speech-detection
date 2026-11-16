"""Streamlit front-end.

    streamlit run app/streamlit_app.py

The app is a thin layer over ``src.inference.Predictor`` and
``src.explainability.lime_explainer`` - there is no model logic here.
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:          # `streamlit run` puts app/ on sys.path, not the root
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.constants import ID2LABEL, LABEL_NAMES  # noqa: E402
from src.inference.predictor import Predictor, available_models  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402

CLASS_STYLE = {
    "Hate speech": {"color": "#B91C1C", "bg": "#FEF2F2", "note": "Attacks or demeans a group. Route to a human reviewer."},
    "Offensive language": {"color": "#B45309", "bg": "#FFFBEB", "note": "Profane or insulting, but not classified as hate speech."},
    "Neither": {"color": "#047857", "bg": "#ECFDF5", "note": "No hateful or offensive content detected."},
}
MODEL_TITLES = {
    "bert": "BERT (fine-tuned)", "cnn": "TextCNN", "bilstm": "BiLSTM",
    "logistic_regression": "TF-IDF + Logistic Regression", "linear_svm": "TF-IDF + Linear SVM",
    "naive_bayes": "TF-IDF + Naive Bayes",
}

st.set_page_config(page_title="Hate Speech Detection", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
      #MainMenu, footer {visibility: hidden;}
      .block-container {padding-top: 2rem; max-width: 1150px;}
      .hero {background: linear-gradient(120deg, #312E81 0%, #4338CA 60%, #6366F1 100%);
             border-radius: 16px; padding: 28px 32px; color: #FFFFFF; margin-bottom: 18px;}
      .hero h1 {color: #FFFFFF; font-size: 1.9rem; margin: 0 0 6px 0; padding: 0;}
      .hero p {color: #E0E7FF; margin: 0; font-size: 1rem;}
      .card {background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 14px; padding: 18px 20px;
             margin-bottom: 14px; box-shadow: 0 1px 2px rgba(17,24,39,0.04);}
      .card h4 {margin: 0 0 10px 0; font-size: 0.8rem; letter-spacing: .06em; text-transform: uppercase; color: #6B7280;}
      .verdict {font-size: 1.6rem; font-weight: 700; margin: 0;}
      .muted {color: #6B7280; font-size: 0.9rem;}
      .bar-row {display: flex; align-items: center; gap: 10px; margin: 8px 0;}
      .bar-label {width: 150px; font-size: 0.9rem; color: #374151;}
      .bar-track {flex: 1; background: #F3F4F6; border-radius: 999px; height: 12px; overflow: hidden;}
      .bar-fill {height: 100%; border-radius: 999px;}
      .bar-value {width: 52px; text-align: right; font-variant-numeric: tabular-nums; font-size: 0.9rem;}
      .tokens {line-height: 2.1; font-size: 1.02rem;}
      .tok {padding: 3px 5px; border-radius: 6px;}
      .notice {border-left: 4px solid #4338CA; background: #EEF2FF; padding: 12px 16px; border-radius: 8px;
               color: #312E81; font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model...")
def get_predictor(model_name: str) -> Predictor:
    return Predictor(model_name)


@st.cache_data
def get_config() -> dict:
    return load_config()


def title_of(model_name: str) -> str:
    return MODEL_TITLES.get(model_name, model_name)


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha:.2f})"


def render_probabilities(probabilities: dict[str, float]) -> str:
    rows = []
    for name in LABEL_NAMES:
        value = probabilities[name]
        rows.append(
            f'<div class="bar-row"><div class="bar-label">{name}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{100 * value:.1f}%;'
            f'background:{CLASS_STYLE[name]["color"]};"></div></div>'
            f'<div class="bar-value">{100 * value:.1f}%</div></div>'
        )
    return "".join(rows)


def render_highlighted(text: str, weights: list[tuple[str, float]], color: str) -> str:
    """Colour each word by its LIME weight. User text is HTML-escaped before rendering."""
    lookup = dict(weights)
    largest = max((abs(w) for w in lookup.values()), default=0.0) or 1.0
    parts = []
    for piece in re.split(r"(\W+)", text):
        safe = html.escape(piece)
        weight = lookup.get(piece)
        if weight is None:
            parts.append(safe)
            continue
        strength = 0.15 + 0.6 * abs(weight) / largest
        background = _rgba(color, strength) if weight > 0 else _rgba("#64748B", strength)
        parts.append(f'<span class="tok" style="background:{background};" title="{weight:+.3f}">{safe}</span>')
    return f'<div class="tokens">{"".join(parts)}</div>'


# ------------------------------------------------------------------ sidebar
cfg = get_config()
models = available_models(cfg)

with st.sidebar:
    st.markdown("### Settings")
    if models:
        model_name = st.selectbox("Model", models, format_func=title_of)
    else:
        model_name = None
        st.warning("No trained model found.")
    explain = st.toggle("Explain prediction (LIME)", value=True)
    num_samples = st.slider("LIME samples", 100, 2000, 300 if model_name == "bert" else 1000, step=100,
                            help="More samples = more stable explanation, but slower.")
    st.markdown("---")
    st.caption("Decision-support tool for content screening. English only. "
               "Predictions can be wrong and must not be the sole basis for moderation decisions.")

# --------------------------------------------------------------------- hero
st.markdown(
    '<div class="hero"><h1>Context-Aware Hate Speech Detection</h1>'
    "<p>An explainable transformer-based NLP system that classifies social media text as "
    "hate speech, offensive language or neither.</p></div>",
    unsafe_allow_html=True,
)

tab_analyze, tab_performance, tab_method, tab_responsible = st.tabs(
    ["Analyze text", "Model performance", "Methodology", "Responsible AI"]
)

# ------------------------------------------------------------------ analyze
with tab_analyze:
    if model_name is None:
        st.info("Train a model first, then reload this page.")
        st.code(
            "python -m src.data.download\npython -m src.data.split\n"
            "python -m src.training.train_baselines\npython -m src.training.train_bert",
            language="powershell",
        )
    else:
        left, right = st.columns([5, 6], gap="large")
        with left:
            text = st.text_area("Text to analyze", height=170, max_chars=1000,
                                placeholder="Paste a post or comment here...")
            submitted = st.button("Analyze", type="primary")
            st.markdown(f'<p class="muted">Model: {title_of(model_name)}</p>', unsafe_allow_html=True)

        if submitted:
            if not text.strip():
                st.session_state.pop("result", None)
                with left:
                    st.warning("Enter some text first.")
            else:
                try:
                    predictor = get_predictor(model_name)
                    result = predictor.predict(text)
                    result["text"] = text
                    result["weights"] = None
                    if explain:
                        from src.explainability.lime_explainer import explain_prediction

                        with st.spinner("Computing explanation..."):
                            result["weights"] = explain_prediction(
                                predictor, text, cfg["explainability"]["lime_num_features"],
                                num_samples, cfg["project"]["seed"],
                            )["weights"]
                    st.session_state["result"] = result
                except ImportError as exc:
                    st.error(f"A required package is missing: {exc}. Run `pip install -r requirements.txt`.")
                except Exception as exc:  # noqa: BLE001 - show a readable message instead of a traceback
                    st.error(f"Prediction failed: {exc}")

        result = st.session_state.get("result")
        with right:
            if not result:
                st.markdown('<div class="card"><h4>Result</h4><p class="muted">The prediction, class '
                            "probabilities and explanation appear here.</p></div>", unsafe_allow_html=True)
            else:
                style = CLASS_STYLE[result["label"]]
                st.markdown(
                    f'<div class="card" style="background:{style["bg"]};border-color:{style["color"]}33;">'
                    f'<h4>Predicted category</h4><p class="verdict" style="color:{style["color"]};">{result["label"]}</p>'
                    f'<p class="muted">{style["note"]} Confidence {100 * result["confidence"]:.1f}% '
                    f'({title_of(result["model"])}).</p></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<div class="card"><h4>Class probabilities</h4>{render_probabilities(result["probabilities"])}</div>',
                            unsafe_allow_html=True)
                if result["weights"]:
                    st.markdown(
                        '<div class="card"><h4>Explanation - words that moved the prediction</h4>'
                        + render_highlighted(result["text"], result["weights"], style["color"])
                        + f'<p class="muted" style="margin-top:10px;">Coloured = pushes towards "{result["label"]}"; '
                        "grey = pushes away. Hover a word for its weight.</p></div>",
                        unsafe_allow_html=True,
                    )
                    weights_table = pd.DataFrame(result["weights"], columns=["Word", "Weight"])
                    st.dataframe(weights_table.style.format({"Weight": "{:+.3f}"}), hide_index=True)
                    st.caption("LIME is an approximation: it shows how the output changes when words are removed, "
                               "not the model's actual reasoning. Weights vary slightly between runs.")

# -------------------------------------------------------------- performance
with tab_performance:
    reports_dir = resolve_path(cfg["paths"]["reports_dir"])
    figures_dir = resolve_path(cfg["paths"]["figures_dir"])
    metrics_dir = resolve_path(cfg["paths"]["metrics_dir"])
    comparison_path = reports_dir / "model_comparison.csv"

    if not comparison_path.is_file():
        st.info("No evaluation results yet. Train the models, then run `python -m src.evaluation.evaluate`.")
    else:
        table = pd.read_csv(comparison_path)
        st.markdown("#### Test-set comparison")
        st.caption("Sorted by macro-F1, the primary metric: it weights the rare hate-speech class "
                   "as much as the frequent offensive class, which accuracy does not.")
        numeric = [c for c in table.columns if table[c].dtype.kind == "f"]
        st.dataframe(table.style.format({c: "{:.3f}" for c in numeric}), hide_index=True)
        if "smoke_test" in table.columns and table["smoke_test"].any():
            st.warning("Rows marked smoke_test were trained on a small subset and are not real results.")
        if (figures_dir / "model_comparison.png").is_file():
            st.image(str(figures_dir / "model_comparison.png"))

        evaluated = [m for m in table["model"] if (metrics_dir / f"{m}.json").is_file()]
        if evaluated:
            chosen = st.selectbox("Details for", evaluated, format_func=title_of)
            with open(metrics_dir / f"{chosen}.json", "r", encoding="utf-8") as fh:
                details = json.load(fh)
            per_class = pd.DataFrame(details["test"]["per_class"]).T.reset_index(names="Class")
            st.dataframe(per_class.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}",
                                                 "support": "{:.0f}"}), hide_index=True)
            for figure in (f"confusion_{chosen}.png", f"history_{chosen}.png", f"robustness_{chosen}.png"):
                if (figures_dir / figure).is_file():
                    st.image(str(figures_dir / figure))
    if (figures_dir / "class_distribution.png").is_file():
        st.markdown("#### Dataset class distribution")
        st.image(str(figures_dir / "class_distribution.png"))

# -------------------------------------------------------------- methodology
with tab_method:
    st.markdown(
        """
#### Pipeline
1. **Data** - Davidson et al. (2017): about 25k English tweets, each labelled by at least three
   crowd workers as *hate speech*, *offensive language* or *neither*.
2. **Cleaning** - user handles and URLs are masked, HTML entities decoded, retweet markers removed.
   BERT receives lightly cleaned text; TF-IDF, CNN and BiLSTM receive lower-cased, de-punctuated text.
3. **Leakage control** - duplicates are removed *before* a stratified 80/10/10 split. The TF-IDF
   vocabulary, the word vocabulary and all weights are fitted on the training split only.
4. **Models** - TF-IDF with Logistic Regression, Linear SVM and Naive Bayes; a TextCNN; a BiLSTM;
   and fine-tuned `bert-base-uncased` as the main model.
5. **Imbalance** - class-weighted losses, and model selection on validation **macro-F1**, not accuracy.
6. **Evaluation** - the test split is used once per model, after selection on validation.
7. **Explainability** - LIME word attributions for every model through one shared interface.
8. **Robustness and bias** - typo, leetspeak and padding perturbations; false-positive rates for
   text that mentions identity groups.

#### Why BERT is "context-aware"
TF-IDF sees a bag of n-grams, and a CNN sees short windows. BERT's self-attention lets every token
attend to every other token in both directions, so the representation of a word depends on the whole
sentence - who is being addressed, negation, quotation.
        """
    )

# ----------------------------------------------------------- responsible AI
with tab_responsible:
    st.markdown(
        '<div class="notice"><b>Disclaimer.</b> This is a research and decision-support prototype. '
        "It is not an automated moderation system. Any action against content or a user should be "
        "reviewed by a person who can see the full context.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
#### Known limitations
- **Labels are subjective.** Crowd annotators often disagree on where *offensive* ends and *hate* begins;
  the model inherits their judgement, including their mistakes.
- **Sampling bias.** The tweets were collected with a lexicon of hateful terms, so the data over-represents
  explicit slurs. Expect the model to lean on trigger words and to miss implicit or coded hate.
- **Dialect and identity bias.** Published audits of this dataset found that classifiers trained on it flag
  tweets written in African-American English more often. Text that merely mentions an identity group, and
  reclaimed in-group usage, can be flagged wrongly.
- **Context is missing.** A single post has no thread, speaker identity, quotation marks or sarcasm cues.
  Counter-speech that quotes hate speech can be classified as hate speech.
- **Scope.** English tweets from around 2017. Language, slang and platforms drift; other languages are unsupported.
- **Evasion.** Misspellings, symbols and padding can change predictions; see the robustness report.
- **Explanations are approximate.** LIME highlights correlate with the model's behaviour but do not prove
  why it decided, and do not make a prediction correct.

#### Intended use
Prioritising content for human review, research and education.

#### Out of scope
Fully automated removal or banning, profiling individuals, legal determinations.

Classes: {", ".join(ID2LABEL.values())}.
        """
    )
