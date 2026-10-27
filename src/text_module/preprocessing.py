import re
from typing import List


def _ensure_punkt():
    try:
        import nltk
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        import nltk
        nltk.download("punkt", quiet=True)
    try:
        import nltk
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        import nltk
        nltk.download("punkt_tab", quiet=True)


def normalize(text: str) -> str:
    """Clean text for display/scoring. Preserves case so NER works correctly."""
    if not text or not text.strip():
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", "[URL]", text)
    return text


def normalize_lower(text: str) -> str:
    """Lowercase normalisation for non-NER tasks (embedding similarity, etc.)."""
    return normalize(text).lower()


def split_sentences(text: str) -> List[str]:
    """Split into sentences. Preserves case so NER picks up named entities correctly."""
    _ensure_punkt()
    from nltk.tokenize import sent_tokenize
    cleaned = normalize(text)
    sents = sent_tokenize(cleaned)
    return [s.strip() for s in sents if len(s.strip()) > 10]
