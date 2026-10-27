from typing import Any, Dict, List, Tuple

from .claim_extraction import extract_claims_simple
from .fact_verification import verify_claim_against_kg
from .ner import get_entities
from .preprocessing import normalize


def _gather_evidence(claim: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Collect entities and evidence for one claim.

    Returns (entities, evidence). Evidence mixes two shapes, both understood by
    the verifier:
        {"subject","relation","object"}   curated KG triple
        {"text","source","url"}           Wikipedia sentence

    NER runs ONCE here and the entities are returned to the caller. The
    previous version called get_entities() again in run_text_pipeline, paying
    for a full BERT forward pass twice per claim.
    """
    try:
        entities = get_entities(claim)
    except Exception:
        entities = []

    evidence: List[Dict] = []

    # --- curated triples: fast, offline, high precision --------------------
    try:
        from src.graph_module.graph_query import query_entity_facts

        seen = set()
        for e in entities:
            for f in query_entity_facts(e["text"]):
                key = (f.get("subject"), f.get("relation"), f.get("object"))
                if key in seen:
                    continue
                seen.add(key)
                f.setdefault("subject", e["text"])
                evidence.append(f)
    except Exception:
        pass

    # --- live Wikipedia: broad coverage, always current --------------------
    # Failures here are non-fatal but must never be silent. An earlier version
    # swallowed them, so a broken retrieval path looked identical to "this
    # claim genuinely has no evidence" — six eval failures with no clue why.
    try:
        enabled = True
        timeout = 6.0
        try:
            from src.config.settings import settings

            enabled = bool(getattr(settings, "enable_wiki_evidence", True))
            timeout = float(getattr(settings, "wiki_timeout", 6.0))
        except Exception as exc:
            _warn(f"settings unavailable, using defaults: {exc}")

        if enabled:
            from src.graph_module.wiki_source import gather_evidence

            names = [e["text"] for e in entities if len(str(e.get("text", ""))) >= 3]
            found = gather_evidence(claim, names, timeout=timeout)
            evidence.extend(found)
            if not found:
                _warn(f"no Wikipedia evidence for {claim[:60]!r} (entities={names})")
    except Exception as exc:
        _warn(f"Wikipedia retrieval failed for {claim[:60]!r}: {exc!r}")

    return entities, evidence


def _warn(message: str) -> None:
    try:
        from loguru import logger

        logger.warning(f"[evidence] {message}")
    except Exception:
        print(f"[evidence] {message}")


# Backwards-compatible alias — test_kg.py and older callers import this name
def _get_kg_facts_for_claim(claim: str) -> List[Dict]:
    return _gather_evidence(claim)[1]


def run_text_pipeline(text: str, max_claims: int = 15) -> Dict[str, Any]:
    """
    Full text analysis pipeline.

    claim extraction -> NER -> evidence retrieval -> NLI verdict
                     -> optional LLM adjudication over the same evidence

    The LLM (when available) overrides NLI because it can reason over the
    whole evidence set at once — closed-world absence, multi-hop links, and
    malformed input are all things a pairwise cross-encoder cannot handle.
    When no LLM is reachable the NLI verdict stands and nothing breaks.
    """
    text = normalize(text)
    if not text:
        return {
            "credibility_score": 0.5,
            "claims": [],
            "explanation": "No text provided.",
            "supported_count": 0,
            "refuted_count": 0,
            "nei_count": 0,
        }

    # --- interpret what the user meant BEFORE retrieving anything ----------
    # Real users type fragments, questions and typos. A fragment has no truth
    # value, so feeding raw sentence splits to the verifier guarantees
    # NOT_ENOUGH_INFO. Normalise first.
    from .query_understanding import understand

    interp = understand(text)
    claims = interp.get("claims") or []

    if not claims:
        return {
            "credibility_score": 0.5,
            "claims": [],
            "explanation": interp.get("clarification")
            or "No verifiable factual claim was found in this text.",
            "supported_count": 0,
            "refuted_count": 0,
            "nei_count": 0,
            "interpretation": interp,
            "checkable": False,
        }

    claims = claims[:max_claims]

    # Resolve the LLM backend once rather than per claim
    llm_enabled = False
    llm_cap = 8
    llm_timeout = 45.0
    try:
        from src.config.settings import settings
        from .llm_judge import detect_backend

        llm_cap = int(getattr(settings, "llm_max_claims", 8))
        llm_timeout = float(getattr(settings, "llm_timeout", 45.0))
        llm_enabled = detect_backend() != "none"
    except Exception:
        llm_enabled = False

    results = []
    supported = refuted = nei = 0
    judged_by = set()

    for idx, c in enumerate(claims):
        entities, evidence = _gather_evidence(c)
        verification = verify_claim_against_kg(c, evidence)
        verification["judge"] = "nli"

        # --- LLM adjudication ---------------------------------------------
        if llm_enabled and idx < llm_cap:
            try:
                from .llm_judge import adjudicate

                llm = adjudicate(c, evidence, timeout=llm_timeout)
                if llm:
                    verification = {
                        **verification,
                        "label": llm["label"],
                        "score": llm["score"],
                        "reason": llm["reason"],
                        "restated_claim": llm.get("restated_claim"),
                        "correction": llm.get("correction"),
                        "judge": llm["judge"],
                        "nli_label": verification.get("label"),
                    }
            except Exception:
                pass

        judged_by.add(verification.get("judge", "nli"))
        results.append({"claim": c, "entities": entities, "verification": verification})

        label = verification["label"]
        if label == "SUPPORTED":
            supported += 1
        elif label == "REFUTED":
            refuted += 1
        else:
            nei += 1

    n = len(results) or 1
    # SUPPORTED = 1.0, NOT_ENOUGH_INFO = 0.5, REFUTED = 0.0
    weighted = (supported * 1.0 + nei * 0.5 + refuted * 0.0) / n
    score = round(float(weighted), 4)

    judge_note = ""
    if judged_by and judged_by != {"nli"}:
        judge_note = " Adjudicated by LLM over retrieved evidence."

    explanation = (
        f"{supported}/{n} claims supported, "
        f"{refuted}/{n} refuted, "
        f"{nei}/{n} inconclusive.{judge_note}"
    )

    return {
        "credibility_score": score,
        "claims": results,
        "explanation": explanation,
        "supported_count": supported,
        "refuted_count": refuted,
        "nei_count": nei,
        "judge": sorted(judged_by),
        "interpretation": interp,   # what the system understood the user to mean
        "checkable": True,
    }
