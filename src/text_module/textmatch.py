"""
Shared fuzzy text matching.

Why this exists
---------------
Wikipedia's search endpoint happily resolves "Barrack Obamma" to the Barack
Obama page. But the retrieved article says "Barack Obama", and an EXACT token
comparison against the user's misspelling scores zero overlap — so every
sentence gets filtered out and the claim comes back NOT_ENOUGH_INFO.

Typo tolerance has to survive the whole pipeline, not just entity lookup.
These helpers treat two tokens as matching when they are near-identical:

    barrack ~ barack   (0.92)
    obamma  ~ obama    (0.91)
    borne   ~ born     (0.89)

Threshold and length gates are tuned to reject look-alike DIFFERENT entities,
which are far more damaging than a missed typo — matching "India" to "Indiana"
would retrieve evidence about the wrong place entirely:

    accepted   barrack~barack 0.92 · obamma~obama 0.91 · borne~born 0.89
    rejected   india~indiana 0.83 · austria~australia 0.88 · kenya~hawaii 0.22

Known limitation: "founded"~"funded" (0.92) still matches. Both are ordinary
verbs rather than entity names, and this only affects which sentences reach the
NLI model — the verdict itself is decided by NLI, not by this heuristic.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Iterable, Set

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of",
    "in", "on", "at", "to", "for", "and", "or", "but", "not", "never", "no",
    "by", "with", "from", "as", "that", "this", "it", "its", "has", "have",
    "had", "he", "she", "they", "them", "his", "her", "their", "who", "which",
    "also", "when", "were", "than", "then", "there", "here", "into", "about",
}

_FUZZY_THRESHOLD = 0.88


def tokens(text: str, min_len: int = 3) -> Set[str]:
    """Lowercased content words, punctuation stripped, stopwords removed."""
    out = set()
    for raw in str(text).lower().split():
        w = raw.strip(".,;:!?\"'()[]{}—–_/\\")
        if w and w not in _STOP and len(w) >= min_len:
            out.add(w)
    return out


def _close(a: str, b: str) -> bool:
    """
    Near-identical words. Cheap gates run before the expensive ratio.

    A real typo is almost always the same length ±1 and keeps the first letter.
    Allowing ±2 let "india" reach "indiana" and "austria" reach "australia".
    """
    if a == b:
        return True

    delta = abs(len(a) - len(b))
    if delta > 1:
        return False
    if len(a) < 4 or len(b) < 4:   # too short to distinguish typo from different word
        return False
    if a[0] != b[0]:               # typos rarely change the first letter
        return False

    return SequenceMatcher(None, a, b).ratio() >= _FUZZY_THRESHOLD


def fuzzy_overlap(claim_tokens: Iterable[str], other_tokens: Iterable[str]) -> int:
    """Count claim tokens that appear in `other_tokens`, exactly or near-exactly."""
    claim_set = set(claim_tokens)
    other_set = set(other_tokens)
    if not claim_set or not other_set:
        return 0

    matched = len(claim_set & other_set)

    # Only the leftovers need the expensive comparison
    for c in claim_set - other_set:
        if any(_close(c, o) for o in other_set):
            matched += 1
    return matched


def coverage(claim_tokens: Iterable[str], other_tokens: Iterable[str]) -> float:
    """Fraction of the claim's content words present in the other text, 0..1."""
    claim_set = set(claim_tokens)
    if not claim_set:
        return 0.0
    return fuzzy_overlap(claim_set, other_tokens) / len(claim_set)
