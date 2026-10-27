"""
Fact verification using NLI (Natural Language Inference).

Uses cross-encoder/nli-deberta-v3-small to decide whether a KG fact ENTAILS or
CONTRADICTS a claim. Unlike cosine similarity, NLI handles negation correctly:
"Trump was never President" is a *contradiction* of "Trump was President", even
though the two sentences are ~95% lexically similar.

Label indices for this model:
    0 = contradiction, 1 = entailment, 2 = neutral

------------------------------------------------------------------------------
DESIGN NOTE — why this is more than "take the max contradiction"
------------------------------------------------------------------------------
A claim pulls in every fact attached to every entity it mentions. A claim about
Elon Musk can therefore be scored against a fact about Bill Gates. Small NLI
models frequently label two unrelated-but-similarly-shaped sentences as
CONTRADICTION with high confidence.

The naive rule (max contradiction across all facts) then fails badly:

    claim: "Elon Musk founded SpaceX."
    facts: "Elon Musk founded SpaceX"                    -> entail  ~1.00
           "Bill Gates funds Gates Foundation"           -> contra  ~1.00  (spurious)
    naive verdict: REFUTED                                       ^^^ wrong

Three rules fix it:

  1. RELEVANCE GATE — a fact may only refute a claim if it actually talks about
     something the claim mentions. Zero content-word overlap means zero vote.
  2. ENTAILMENT WINS TIES — if the graph literally contains the claim, that is
     direct positive evidence and outranks a contradiction from another fact.
  3. MARGIN — contradiction must beat the best entailment by a clear margin
     before it can flip the verdict, instead of winning by a rounding error.
"""

from typing import Any, Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Model singletons
# ---------------------------------------------------------------------------
_nli_model = None
_st_model = None  # kept so health-check / warmup imports don't break

_CONTRADICTION = 0
_ENTAILMENT = 1
_NEUTRAL = 2

_SUPPORT_THRESH = 0.60   # entailment prob needed to call SUPPORTED
_REFUTE_THRESH = 0.60    # contradiction prob needed to call REFUTED
_MARGIN = 0.12           # contradiction must exceed entailment by this much

# Words that carry no discriminating signal when matching a fact to a claim
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of",
    "in", "on", "at", "to", "for", "and", "or", "but", "not", "never", "no",
    "by", "with", "from", "as", "that", "this", "it", "its", "has", "have",
    "had", "he", "she", "they", "them", "his", "her", "their", "who", "which",
    "-", "–", "—",
}


def _get_nli_model():
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder

        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    return _nli_model


def _get_st_model():
    """Legacy — kept so health check / warmup imports don't break."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer

        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fact_to_text(f: dict) -> str:
    """
    Turn an evidence item into a premise sentence.

    Two shapes are supported:
      * {"text": "..."}                       — a Wikipedia sentence, used as-is
      * {"subject", "relation", "object"}     — a KG triple, joined into prose
    """
    if f.get("text"):
        return str(f["text"]).strip()

    subj = str(f.get("subject", "") or "")
    rel = str(f.get("relation", "") or "").replace("_", " ").lower()
    obj = str(f.get("object", "") or "")
    parts = [p for p in (subj, rel, obj) if p]
    return " ".join(parts).strip() or str(f)


def _tokens(text: str) -> set:
    """Content words, lowercased, punctuation stripped."""
    out = set()
    for raw in str(text).lower().split():
        w = raw.strip(".,;:!?\"'()[]{}")
        if w and w not in _STOP and len(w) > 1:
            out.add(w)
    return out


def _relevance(fact: dict, claim_tokens: set) -> float:
    """
    How much this evidence is 'about' the claim. 0.0 means unrelated, so it
    gets no vote on the verdict.

    Free-text evidence (Wikipedia) is scored on how much of the CLAIM it
    covers; a triple is scored on how much of its own entities the claim
    mentions. Scoring a long sentence the second way would unfairly punish it
    for containing extra words.
    """
    from .textmatch import coverage, fuzzy_overlap

    if fact.get("text"):
        # Fuzzy so misspelled claims still match correctly-spelled evidence
        return coverage(claim_tokens, _tokens(fact["text"]))

    entity_tokens = _tokens(fact.get("subject", "")) | _tokens(fact.get("object", ""))
    if not entity_tokens:
        return 0.0
    return fuzzy_overlap(entity_tokens, claim_tokens) / len(entity_tokens)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _nei(reason: str, best_fact=None) -> Dict[str, Any]:
    return {
        "label": "NOT_ENOUGH_INFO",
        "score": 0.5,
        "reason": reason,
        "best_fact": best_fact,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def verify_claim_against_kg(
    claim: str,
    kg_facts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Verify a claim against KG facts.

    Returns {label, score, reason, best_fact} where label is one of
    SUPPORTED | REFUTED | NOT_ENOUGH_INFO.
    """
    if not kg_facts:
        return _nei("No KG facts found for entities in this claim.")

    claim_tokens = _tokens(claim)

    # --- 1. relevance gate -------------------------------------------------
    scored = [(f, _relevance(f, claim_tokens)) for f in kg_facts]
    relevant = [(f, r) for f, r in scored if r > 0.0]

    if not relevant:
        return _nei(
            f"{len(kg_facts)} KG fact(s) retrieved, but none overlap with this claim."
        )

    # keep the strongest overlaps; caps NLI calls and drops weak noise
    relevant.sort(key=lambda t: t[1], reverse=True)
    relevant = relevant[:12]

    facts = [f for f, _ in relevant]
    rels = [r for _, r in relevant]
    fact_texts = [_fact_to_text(f) for f in facts]

    # --- 2. NLI over (premise=fact, hypothesis=claim) ----------------------
    model = _get_nli_model()
    logits = model.predict([(t, claim) for t in fact_texts], apply_softmax=False)
    logits = np.asarray(logits)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    probs = _softmax(logits)

    entail = probs[:, _ENTAILMENT]
    contra = probs[:, _CONTRADICTION]

    # Weight by relevance so a marginally-related fact cannot dominate
    weighted_entail = entail * (0.55 + 0.45 * np.asarray(rels))
    weighted_contra = contra * (0.55 + 0.45 * np.asarray(rels))

    e_idx = int(np.argmax(weighted_entail))
    c_idx = int(np.argmax(weighted_contra))
    best_e = float(entail[e_idx])
    best_c = float(contra[c_idx])

    def _cite(idx: int) -> dict:
        """Attach provenance so the UI can show where the evidence came from."""
        f = facts[idx]
        return {
            "source": f.get("source") or "Knowledge graph",
            "source_url": f.get("url"),
        }

    # --- 3. decide ---------------------------------------------------------
    # Entailment wins ties: evidence that directly states the claim is stronger
    # than a contradiction inferred from some other sentence.
    if best_e >= _SUPPORT_THRESH and best_e >= best_c - 0.05:
        return {
            "label": "SUPPORTED",
            "score": round(best_e, 4),
            "reason": f"Entailed by evidence (confidence {best_e:.0%})",
            "best_fact": fact_texts[e_idx],
            **_cite(e_idx),
        }

    if best_c >= _REFUTE_THRESH and best_c > best_e + _MARGIN:
        return {
            "label": "REFUTED",
            "score": round(1.0 - best_c, 4),
            "reason": f"Contradicted by evidence (confidence {best_c:.0%})",
            "best_fact": fact_texts[c_idx],
            **_cite(c_idx),
        }

    lead = e_idx if best_e >= best_c else c_idx
    out = _nei(
        f"Inconclusive (entail {best_e:.0%}, contradict {best_c:.0%})",
        fact_texts[lead],
    )
    out.update(_cite(lead))
    return out
