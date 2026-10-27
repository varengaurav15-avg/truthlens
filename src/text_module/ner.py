"""
Named entity recognition.

CASING MATTERS — a lot.
----------------------
dslim/bert-base-NER is trained on CoNLL-2003, which is properly capitalised.
Feed it lowercase text and recall collapses:

    "Osama the president of USA"  -> PER: Osama,  LOC: USA
    "osama the president of usa"  -> (nothing)

Users type lowercase constantly. Without handling it, the whole pipeline goes
silent: no entities -> no evidence retrieved -> every claim returns
NOT_ENOUGH_INFO and scores a flat 50%.

So when the input looks uncased and NER finds little, we retry on a
title-cased copy and map the results back onto the original text.
"""

import re
from typing import Any, Dict, List

_ner_pipeline = None

# Words that should stay lowercase when we synthesise capitalisation
_MINOR = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "by", "with", "from", "as",
    "that", "this", "it", "its", "has", "have", "had", "not", "never", "no",
}

# Common shorthand the model will not resolve on its own
_EXPAND = {
    "usa": "United States",
    "us": "United States",
    "uk": "United Kingdom",
    "uae": "United Arab Emirates",
    "eu": "European Union",
    "un": "United Nations",
    "who": "World Health Organization",
    "nasa": "NASA",
    "fbi": "FBI",
    "cia": "CIA",
    "pm": "Prime Minister",
    "potus": "President of the United States",
}


def _get_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        from transformers import pipeline

        _ner_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
        )
    return _ner_pipeline


def _looks_uncased(text: str) -> bool:
    """True when the text has essentially no capital letters."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    caps = sum(1 for c in letters if c.isupper())
    return caps / len(letters) < 0.03


def _title_case(text: str) -> str:
    """Capitalise everything except minor words — a crude but effective truecase."""
    out = []
    for i, word in enumerate(text.split()):
        low = word.lower()
        if i > 0 and low in _MINOR:
            out.append(low)
        else:
            out.append(word[:1].upper() + word[1:] if word else word)
    return " ".join(out)


# Words that are never a useful entity on their own. Without this, NER tagging
# "New Movie Named" as MISC matched "New Delhi" in the knowledge graph on the
# shared token "new", and a claim about a film retrieved the capital of India.
_GENERIC = {
    # content words too vague to identify an entity
    "new", "movie", "film", "named", "name", "video", "song", "album", "show",
    "series", "story", "news", "report", "photo", "picture", "image", "clip",
    "post", "article", "today", "yesterday", "tomorrow", "year", "day", "time",
    "people", "man", "woman", "person", "thing", "place", "part", "way",
    # function words — these appear capitalised at the start of a sentence and
    # would otherwise be extracted as entities ("The meeting has been...")
    "the", "this", "that", "there", "then", "they", "these", "those",
    "was", "were", "has", "have", "had", "been", "and", "but", "for",
    "with", "from", "into", "about", "his", "her", "its", "our", "their",
    "who", "what", "when", "where", "why", "how", "not", "never",
    # job titles — a role is not an entity. Bare "President" was being looked
    # up on Wikipedia, wasting a request that "Osama" needed. Multi-word forms
    # like "President of the United States" still pass, because they contain
    # non-generic words.
    "president", "minister", "prime", "chairman", "chairperson", "ceo",
    "founder", "king", "queen", "secretary", "governor", "mayor", "director",
    "captain", "leader", "chief", "officer", "actor", "actress", "singer",
    # calendar words — "Chris Eubank was born on 8 August 1966" was spending an
    # entity slot and a Wikipedia lookup on "August"
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    # nationality adjectives are modifiers, not the subject of a claim
    "american", "british", "indian", "german", "french", "italian", "spanish",
    "chinese", "japanese", "russian", "canadian", "australian", "english",
}


def _run(text: str) -> List[Dict[str, Any]]:
    ner = _get_ner_pipeline()
    try:
        results = ner(text)
    except Exception:
        return []
    out = []
    for r in results:
        word = str(r.get("word", "")).replace("##", "").strip()
        if word:
            out.append({"text": word, "label": r.get("entity_group", "MISC")})
    return out


def _is_usable(entity: str, original: str) -> bool:
    """
    Reject NER output that isn't a real entity.

    Two failure modes this catches:
      * subword fragments — "yash" came back as "Ya" and "sh", neither of which
        is a word in the source text
      * generic noun phrases — "New Movie Named" is not an entity, and matching
        it against a knowledge graph retrieves nonsense
    """
    e = entity.strip().lower()
    if len(e) < 3:
        return False

    # must appear as whole word(s) in the original text
    if not re.search(rf"(?<![a-z0-9]){re.escape(e)}(?![a-z0-9])", original.lower()):
        return False

    # must contain at least one non-generic word
    words = [w for w in re.split(r"[\s\-]+", e) if w]
    return any(w not in _GENERIC and len(w) >= 3 for w in words)


def get_entities(claim: str) -> List[Dict[str, Any]]:
    """
    Extract entities, tolerating lowercase input.

    Also emits expanded forms of common acronyms ("usa" -> "United States") so
    downstream Wikipedia lookups resolve to the right article.
    """
    if not claim or not claim.strip():
        return []

    entities = _run(claim)

    # Retry with synthetic capitalisation when the input is uncased and NER
    # found little — this is the difference between a verdict and a blank 50%.
    if len(entities) < 2 and _looks_uncased(claim):
        entities.extend(_run(_title_case(claim)))

    # Add acronym expansions found anywhere in the text
    lowered = {w.strip(".,;:!?\"'()").lower() for w in claim.split()}
    for short, full in _EXPAND.items():
        if short in lowered:
            entities.append({"text": full, "label": "MISC"})

    # Drop fragments and generic phrases before anything downstream sees them
    entities = [e for e in entities if _is_usable(e["text"], claim)]

    # ALWAYS supplement with capitalised spans, not just when NER returns
    # nothing. NER silently missed "Narendra Modi" in "Narendra Modi is the
    # Prime Minister of Canada" and returned only the country, so neither the
    # knowledge graph nor Wikipedia ever saw the subject of the claim.
    source = claim if not _looks_uncased(claim) else _title_case(claim)
    known = {e["text"].lower() for e in entities}
    spans = []
    for m in re.finditer(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*", source):
        cand = m.group(0).strip()
        low = cand.lower()
        if low in known or not _is_usable(cand, source):
            continue
        # skip spans already contained in a detected entity
        if any(low in k or k in low for k in known):
            continue
        spans.append({"text": cand, "label": "MISC"})

    # Longer spans first — "Narendra Modi" is a better lookup key than "Modi"
    spans.sort(key=lambda e: -len(e["text"]))
    entities.extend(spans)

    # De-duplicate case-insensitively, keeping first occurrence.
    # Capped at 5: every entity costs a Wikipedia round trip.
    seen, unique = set(), []
    for e in entities:
        key = e["text"].lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(e)
    return unique[:5]
