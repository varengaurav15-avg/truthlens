"""
LLM adjudication layer.

WHAT THIS FIXES
---------------
The NLI cross-encoder compares ONE premise to ONE hypothesis. That makes it
fast and cheap, but it cannot do three things this project needs:

  1. Closed-world reasoning.
     "Osama is the president of the USA" is false because he is absent from
     the list of presidents. NLI reasons from a single sentence, so absence
     is invisible to it. An LLM shown the list can reason over the whole set.

  2. Malformed input.
     "osama the president of usa" has no verb. NLI treats it as a fragment.
     An LLM restates it as a proposition before judging.

  3. Multi-hop evidence.
     "Bill Gates founded SpaceX" needs two facts combined — Musk founded
     SpaceX, and founding is effectively exclusive. NLI sees them separately.

DESIGN RULES
------------
* The LLM is GROUNDED. It is shown retrieved evidence and told to use only
  that. It is never asked to recall facts from its own weights, because that
  is how you get confident hallucinations inside a misinformation detector.
* It is OPTIONAL. If no backend is reachable the NLI verdict stands and the
  pipeline behaves exactly as before.
* It returns STRUCTURED output, validated before use. A malformed reply is
  discarded rather than guessed at.

BACKENDS (tried in order)
-------------------------
  ollama       local, free, no key      http://localhost:11434
  transformers local, uses your venv    small instruct model on CPU
  none         disabled                 NLI verdict is final
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

_VALID = {"SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"}

_backend: Optional[str] = None      # resolved once
_hf_pipe = None                      # transformers pipeline cache

SYSTEM_RULES = """You are a strict fact-checking adjudicator.

Decide whether the CLAIM is supported or refuted BY THE EVIDENCE PROVIDED.

Rules:
- Use ONLY the evidence given. Never use outside knowledge.
- Ignore spelling and grammar errors in the claim; judge what it means.
- If the evidence directly states the claim is true -> SUPPORTED
- If the evidence directly conflicts with the claim -> REFUTED
- If the evidence gives a complete list or a single office-holder and the
  claim names someone or something absent from it -> REFUTED
- If the evidence simply does not address the claim -> NOT_ENOUGH_INFO
- Do not guess. NOT_ENOUGH_INFO is a correct answer when evidence is thin.

CRITICAL — fiction is not biography:
- Evidence may describe roles an actor PLAYED, plot summaries, or quotes from
  reviews. Never treat a fictional character's traits as facts about the real
  person. "X plays an aged lawyer in the film" does NOT mean X is a lawyer.
- If the only evidence available is about fictional roles, reviews, or plot,
  answer NOT_ENOUGH_INFO rather than reasoning from it.
- A person being absent from the evidence does not mean they are dead, alive,
  or anything else. Only state what the evidence actually says.

Tense signals life status:
- Wikipedia writes "X IS an actor" for living people, and "X WAS an actor"
  plus a death date for people who have died.
- So present-tense description with no death date is evidence AGAINST a claim
  that someone has died. Treat such a death claim as REFUTED and say the
  evidence describes them in the present tense with no recorded death.

For "correction": if the verdict is REFUTED, write the accurate version of the
claim as a single plain sentence, taken strictly from the evidence. For any
other verdict use an empty string. Never invent a correction.

Reply with ONLY a JSON object, no other text:
{"restated_claim":"<claim as a clear sentence>","verdict":"SUPPORTED|REFUTED|NOT_ENOUGH_INFO","confidence":<0.0-1.0>,"reason":"<one short sentence explaining why>","correction":"<the true statement, or empty string>"}"""


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
# Each preset: (base_url, default_model, openai_compatible)
PRESETS = {
    "groq":   ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", True),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini", True),
    "openrouter": ("https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct", True),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta", "gemini-2.0-flash", False),
    "ollama": ("http://localhost:11434", "llama3.2:3b", False),
}


def _cfg(name: str, default=""):
    try:
        from src.config.settings import settings

        return getattr(settings, name, default)
    except Exception:
        return default


def _provider_cfg(provider: str):
    base, model, _ = PRESETS.get(provider, PRESETS["ollama"])
    return (
        str(_cfg("llm_base_url") or base).rstrip("/"),
        str(_cfg("llm_model") or model),
    )


def _ollama_alive(timeout: float = 1.5) -> bool:
    base, _ = _provider_cfg("ollama")
    try:
        import requests

        return requests.get(f"{base}/api/tags", timeout=timeout).status_code == 200
    except Exception:
        return False


def detect_backend(force: bool = False) -> str:
    """
    Resolve the best available backend. Cached after the first call.

    Priority under "auto": an explicitly configured cloud provider (far more
    accurate) > local Ollama > none. Never raises — a missing LLM just means
    the NLI verdict stands.
    """
    global _backend
    if _backend is not None and not force:
        return _backend

    configured = str(_cfg("llm_provider", "auto") or "auto").lower()
    if configured == "auto":
        legacy = str(_cfg("llm_backend", "auto") or "auto").lower()
        if legacy not in ("auto", ""):
            configured = legacy

    key = str(_cfg("llm_api_key", "")).strip()

    if configured == "none":
        _backend = "none"
    elif configured in ("groq", "openai", "openrouter", "gemini"):
        _backend = configured if key else ("ollama" if _ollama_alive() else "none")
    elif configured == "ollama":
        _backend = "ollama" if _ollama_alive() else "none"
    elif configured == "transformers":
        _backend = "transformers"
    else:  # auto with no explicit provider
        _backend = "ollama" if _ollama_alive() else "none"

    return _backend


def status() -> Dict[str, Any]:
    b = detect_backend()
    model = None
    if b in PRESETS:
        model = _provider_cfg(b)[1]
    elif b == "transformers":
        model = _hf_model_id()

    notes = {
        "groq": "Groq cloud (Llama-3.3-70B) adjudicating.",
        "openai": "OpenAI cloud adjudicating.",
        "openrouter": "OpenRouter cloud adjudicating.",
        "gemini": "Google Gemini adjudicating.",
        "ollama": "Local Ollama adjudicating.",
        "transformers": "Local transformers LLM (slow on CPU).",
        "none": "No LLM reachable — NLI verdict is final.",
    }
    return {
        "backend": b,
        "enabled": b != "none",
        "model": model,
        "cloud": b in ("groq", "openai", "openrouter", "gemini"),
        "note": notes.get(b, ""),
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _hf_model_id() -> str:
    try:
        from src.config.settings import settings

        return str(getattr(settings, "llm_model", "Qwen/Qwen2.5-1.5B-Instruct"))
    except Exception:
        return "Qwen/Qwen2.5-1.5B-Instruct"


def _generate_ollama(prompt: str, timeout: float) -> Optional[str]:
    base, model = _provider_cfg("ollama")
    try:
        import requests

        r = requests.post(
            f"{base}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0, "num_predict": 260},
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception:
        return None


def _generate_openai_compatible(provider: str, prompt: str, timeout: float) -> Optional[str]:
    """Works for Groq, OpenAI, OpenRouter and any OpenAI-shaped endpoint."""
    base, model = _provider_cfg(provider)
    key = str(_cfg("llm_api_key", "")).strip()
    try:
        import requests

        r = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 320,
                "response_format": {"type": "json_object"},
            },
            timeout=timeout,
        )
        if r.status_code == 400:      # some models reject response_format
            r = requests.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 320,
                },
                timeout=timeout,
            )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def _generate_gemini(prompt: str, timeout: float) -> Optional[str]:
    base, model = _provider_cfg("gemini")
    key = str(_cfg("llm_api_key", "")).strip()
    try:
        import requests

        r = requests.post(
            f"{base}/models/{model}:generateContent?key={key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 320,
                    "responseMimeType": "application/json",
                },
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return None


def _generate_transformers(prompt: str) -> Optional[str]:
    global _hf_pipe
    try:
        if _hf_pipe is None:
            from transformers import pipeline

            _hf_pipe = pipeline(
                "text-generation",
                model=_hf_model_id(),
                torch_dtype="auto",
                device_map="cpu",
            )
        out = _hf_pipe(
            [{"role": "user", "content": prompt}],
            max_new_tokens=220,
            do_sample=False,
            return_full_text=False,
        )
        return out[0]["generated_text"]
    except Exception:
        return None


def _extract_json(raw: str) -> Optional[dict]:
    """Pull the first JSON object out of a model reply."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def adjudicate(
    claim: str,
    evidence: List[Dict[str, Any]],
    timeout: float = 45.0,
) -> Optional[Dict[str, Any]]:
    """
    Ask the LLM to judge `claim` against `evidence`.

    Returns {label, score, reason, restated_claim, judge} or None when no
    backend is available or the reply could not be validated — callers must
    keep the NLI verdict in that case.
    """
    backend = detect_backend()
    if backend == "none" or not claim.strip():
        return None

    lines = []
    for i, e in enumerate(evidence[:8], 1):
        text = e.get("text") or " ".join(
            str(e.get(k, "")) for k in ("subject", "relation", "object")
        ).replace("_", " ").strip()
        if not text:
            continue
        src = e.get("source") or "knowledge graph"
        lines.append(f"[{i}] ({src}) {text}")

    evidence_block = "\n".join(lines) if lines else "(no evidence retrieved)"
    prompt = f"{SYSTEM_RULES}\n\nEVIDENCE:\n{evidence_block}\n\nCLAIM: {claim}\n\nJSON:"

    if backend in ("groq", "openai", "openrouter"):
        raw = _generate_openai_compatible(backend, prompt, timeout)
    elif backend == "gemini":
        raw = _generate_gemini(prompt, timeout)
    elif backend == "ollama":
        raw = _generate_ollama(prompt, timeout)
    else:
        raw = _generate_transformers(prompt)
    data = _extract_json(raw or "")
    if not isinstance(data, dict):
        return None

    verdict = str(data.get("verdict", "")).strip().upper().replace(" ", "_")
    if verdict in {"SUPPORT", "TRUE", "SUPPORTED_BY_EVIDENCE"}:
        verdict = "SUPPORTED"
    if verdict in {"REFUTE", "FALSE", "CONTRADICTED"}:
        verdict = "REFUTED"
    if verdict in {"UNVERIFIABLE", "UNKNOWN", "INSUFFICIENT", "NEI"}:
        verdict = "NOT_ENOUGH_INFO"
    if verdict not in _VALID:
        return None

    try:
        conf = float(data.get("confidence", 0.6))
    except Exception:
        conf = 0.6
    conf = max(0.0, min(1.0, conf))

    # Score always means "how credible is this claim", consistent across labels
    if verdict == "SUPPORTED":
        score = round(conf, 4)
    elif verdict == "REFUTED":
        score = round(1.0 - conf, 4)
    else:
        score = 0.5

    # Only surface a correction for refuted claims — anything else would be
    # the model inventing a "fix" for something that wasn't wrong.
    correction = str(data.get("correction", "")).strip()[:300]
    if verdict != "REFUTED":
        correction = ""

    return {
        "label": verdict,
        "score": score,
        "reason": str(data.get("reason", ""))[:300] or f"LLM verdict ({conf:.0%} confidence)",
        "restated_claim": str(data.get("restated_claim", ""))[:300] or None,
        "correction": correction or None,
        "judge": backend,
    }
