"""Text cleaning and normalisation for movie reviews (NLTK based)."""
from __future__ import annotations

import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

HTML_TAG = re.compile(r"<[^>]+>")
NON_ALPHA = re.compile(r"[^a-z\s']")
MULTISPACE = re.compile(r"\s+")

# Negation words must survive stopword removal - they flip sentiment.
KEEP = {"not", "no", "nor", "never", "none", "cannot", "n't", "very", "too", "but"}


def ensure_nltk() -> None:
    for pkg, path in [
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
        ("punkt", "tokenizers/punkt"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


@lru_cache(maxsize=1)
def _tools():
    ensure_nltk()
    stops = set(stopwords.words("english")) - KEEP
    return stops, WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, strip HTML/punctuation, drop stopwords, lemmatise."""
    stops, lemmatizer = _tools()
    text = HTML_TAG.sub(" ", str(text).lower())
    text = text.replace("n't", " not")
    text = NON_ALPHA.sub(" ", text)
    text = MULTISPACE.sub(" ", text).strip()
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in text.split()
        if tok not in stops and len(tok) > 1
    ]
    return " ".join(tokens)


def clean_series(texts) -> list[str]:
    return [clean_text(t) for t in texts]
