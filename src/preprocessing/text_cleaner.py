"""Tweet cleaning.

Two levels, because the model families need different inputs:

* ``light_clean``  -> for BERT. Transformers were pre-trained on natural text and
  have their own sub-word tokenizer, so we only remove Twitter noise and mask
  private/irrelevant tokens (user handles, URLs). Aggressive cleaning would
  destroy context that self-attention can use.

* ``full_clean``   -> for TF-IDF, CNN and BiLSTM. These build their vocabulary
  from our data, so we normalise harder (lowercase, strip punctuation, squash
  "soooo" -> "soo") to keep the vocabulary small and dense.

Deliberately NOT done:
* stop-word removal - pronouns and negations ("you", "they", "not") carry the
  targeting/intent signal that separates hate from merely offensive text;
* stemming/lemmatisation - little gain on tweets, and it merges forms whose
  connotations differ.

Masking @handles also serves privacy: no username reaches a model or a report.
"""
from __future__ import annotations

import html
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
RETWEET_RE = re.compile(r"\bRT\b\s*:?\s*")          # the retweet marker, upper-case only
HASHTAG_RE = re.compile(r"#(\w+)")
ELONGATION_RE = re.compile(r"(.)\1{2,}")            # 3+ repeats of a character
NON_ALNUM_RE = re.compile(r"[^a-z0-9'\s]")
WHITESPACE_RE = re.compile(r"\s+")

BERT_USER_TOKEN = "@user"
BERT_URL_TOKEN = "http"
CLASSIC_USER_TOKEN = "usermention"
CLASSIC_URL_TOKEN = "urllink"


def _strip_twitter_noise(text: object, user_token: str, url_token: str) -> str:
    if text is None:
        return ""
    text = str(text)
    # The CSV stores entities such as &amp; and &#8220; ; decode them first.
    text = html.unescape(text)
    text = text.replace("’", "'").replace("‘", "'")
    text = RETWEET_RE.sub(" ", text)
    text = URL_RE.sub(f" {url_token} ", text)
    text = MENTION_RE.sub(f" {user_token} ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    # "@user @user @user" -> "@user": reply chains add length but no meaning.
    repeated = re.compile(rf"(?:{re.escape(user_token)}\s*){{2,}}")
    text = repeated.sub(f"{user_token} ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def light_clean(text: object) -> str:
    """Minimal cleaning for transformer models (case and punctuation kept)."""
    return _strip_twitter_noise(text, BERT_USER_TOKEN, BERT_URL_TOKEN)


def full_clean(text: object) -> str:
    """Heavier normalisation for bag-of-words and word-embedding models."""
    text = _strip_twitter_noise(text, CLASSIC_USER_TOKEN, CLASSIC_URL_TOKEN)
    text = text.lower()
    text = HASHTAG_RE.sub(r"\1", text)              # "#word" -> "word"
    text = ELONGATION_RE.sub(r"\1\1", text)         # "soooo" -> "soo"
    text = NON_ALNUM_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()
