"""
Query understanding — work out what the user actually means.

WHY
---
Real users of a "is this real or fake?" site do not type clean propositions.
They type things like:

    "yash in a new movie named toxic"          fragment, no verb
    "amitabh bachchan dead??"                  question, no punctuation sense
    "is it true modi resigned"                 question form
    "BREAKING: 5g causes covid share now!!!"   headline with noise
    "hello"                                    not a claim at all

The old pipeline regex-split the input into sentences and fed the fragments
straight to retrieval. A fragment with no verb has no truth value, so the
verifier had nothing to work with and returned NOT_ENOUGH_INFO.

This module puts an interpretation step first: normalise messy input into
explicit, checkable statements BEFORE any retrieval happens. It is the
difference between a tool that demands well-formed input and one that behaves
the way people expect a search box to behave.

It never invents facts. It only restates what the user wrote as a proposition
that can be true or false, and says so when the input isn't checkable at all.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

MAX_CLAIMS = 8

PROMPT = """You interpret what a user wants checked on a fact-checking website.

Users type carelessly: typos, no capitals, no punctuation, fragments, questions,
pasted headlines, ALL CAPS, emoji. Work out the checkable statement behind it.

Return ONLY a JSON object:
{{
  "input_type": "claim" | "question" | "article" | "entity" | "not_checkable",
  "topic": "<3-6 word description of the subject>",
  "interpretation": "<one sentence: what the user is asking, in plain English>",
  "claims": ["<explicit checkable statement>", ...],
  "clarification": "<question to ask the user, or empty string>"
}}

Rules for "claims":
- Rewrite each into a complete statement that can be TRUE or FALSE.
- Fix spelling, capitalisation and grammar. "barrack obamma" -> "Barack Obama".
- Expand shorthand: "pm" -> "Prime Minister", "usa" -> "United States".
- Turn questions into the statement being asked about.
  "did modi resign?" -> "Narendra Modi resigned."
- Keep a fragment's evident meaning; supply only the missing verb.
  "yash in a new movie named toxic" -> "Yash stars in a film titled Toxic."
- Split multi-part input into separate claims, at most {max_claims}.
- NEVER add facts the user did not state. Do not answer the claim. Do not
  guess who is right. You are rewriting, not verifying.
- If the input states no factual proposition (a greeting, an opinion, a
  question about preference, gibberish), set input_type "not_checkable",
  return an empty claims list, and put a helpful next step in "clarification".

Examples:
input: "amitabh bachchan dead??"
{{"input_type":"question","topic":"Amitabh Bachchan death report","interpretation":"The user wants to know whether Amitabh Bachchan has died.","claims":["Amitabh Bachchan has died."],"clarification":""}}

input: "hello"
{{"input_type":"not_checkable","topic":"greeting","interpretation":"The user has not provided anything to check.","claims":[],"clarification":"Paste the news, message or claim you would like verified."}}

USER INPUT:
{text}

JSON:"""


def _fallback(text: str) -> Dict[str, Any]:
    """Regex claim extraction, used when no LLM is reachable."""
    from .claim_extraction import extract_claims_simple

    claims = extract_claims_simple(text, max_claims=MAX_CLAIMS)
    return {
        "input_type": "claim" if claims else "not_checkable",
        "topic": "",
        "interpretation": "",
        "claims": claims,
        "clarification": "" if claims else "Paste the news, message or claim you would like verified.",
        "understood_by": "rules",
    }


def _extract_json(raw: str):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def understand(text: str, timeout: float = 30.0) -> Dict[str, Any]:
    """
    Turn raw user input into checkable claims.

    Always returns a dict with claims/interpretation/clarification. Falls back
    to rule-based extraction if the LLM is unavailable, so the site keeps
    working without one.
    """
    text = (text or "").strip()
    if not text:
        return {
            "input_type": "not_checkable",
            "topic": "",
            "interpretation": "",
            "claims": [],
            "clarification": "Enter the claim, message or article you want checked.",
            "understood_by": "rules",
        }

    try:
        from .llm_judge import detect_backend

        if detect_backend() == "none":
            return _fallback(text)
    except Exception:
        return _fallback(text)

    # Very long input: interpret the opening, which is where the claim lives
    snippet = text[:4000]

    try:
        from .llm_judge import _generate_gemini, _generate_ollama, _generate_openai_compatible, detect_backend

        backend = detect_backend()
        prompt = PROMPT.format(text=snippet, max_claims=MAX_CLAIMS)

        if backend in ("groq", "openai", "openrouter"):
            raw = _generate_openai_compatible(backend, prompt, timeout)
        elif backend == "gemini":
            raw = _generate_gemini(prompt, timeout)
        elif backend == "ollama":
            raw = _generate_ollama(prompt, timeout)
        else:
            return _fallback(text)
    except Exception:
        return _fallback(text)

    data = _extract_json(raw or "")
    if not isinstance(data, dict):
        return _fallback(text)

    claims = [
        str(c).strip()
        for c in (data.get("claims") or [])
        if isinstance(c, (str, int, float)) and str(c).strip()
    ][:MAX_CLAIMS]

    input_type = str(data.get("input_type", "claim")).strip().lower()
    if input_type not in {"claim", "question", "article", "entity", "not_checkable"}:
        input_type = "claim" if claims else "not_checkable"

    # A model that returns no claims but calls it checkable is contradicting
    # itself — trust the claims list and fall back to the rule extractor.
    if not claims and input_type != "not_checkable":
        return _fallback(text)

    return {
        "input_type": input_type,
        "topic": str(data.get("topic", ""))[:120],
        "interpretation": str(data.get("interpretation", ""))[:300],
        "claims": claims,
        "clarification": str(data.get("clarification", ""))[:300],
        "understood_by": "llm",
    }
