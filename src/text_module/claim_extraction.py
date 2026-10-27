from typing import List
import re


_SKIP_PATTERNS = re.compile(
    r"^(however|according to|reportedly|allegedly|it is (said|believed|reported)|"
    r"sources say|experts say|officials say|critics argue|some say|"
    r"meanwhile|furthermore|additionally|in addition|as a result)",
    re.IGNORECASE,
)

_CLAIM_MARKERS = re.compile(
    r"\b(is|are|was|were|has|have|had|will|would|can|could|"
    r"confirmed|denied|announced|revealed|declared|stated|"
    r"caused|led to|resulted in|killed|injured|arrested|"
    r"increased|decreased|rose|fell|launched|signed|passed)\b",
    re.IGNORECASE,
)


def extract_claims_simple(text: str, max_claims: int = 20) -> List[str]:
    """
    Extract verifiable factual claims from text.

    1. Split into sentences (case-preserved so NER works correctly).
    2. Filter out navigational / hedged openers.
    3. Prefer sentences with strong claim markers.
    4. Fall back to all remaining sentences if none qualify.
    """
    from .preprocessing import split_sentences

    sents = split_sentences(text)
    if not sents:
        return []

    strong, weak = [], []
    for s in sents:
        if _SKIP_PATTERNS.match(s.strip()):
            continue
        if len(s.strip()) < 15:
            continue
        if s.count(",") > max(3, len(s) // 40):
            continue
        if _CLAIM_MARKERS.search(s):
            strong.append(s.strip())
        else:
            weak.append(s.strip())

    candidates = strong + weak
    return candidates[:max_claims] if candidates else sents[:max_claims]
