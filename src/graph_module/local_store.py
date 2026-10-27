"""
Offline knowledge-graph store.

Reads data/kg_facts.json and answers the same queries as the Neo4j backend,
with no network, no server, and no credentials.

Why this exists
---------------
Neo4j Aura's free tier pauses an instance after 3 days of inactivity and
DELETES it after 30. When that happens the hostname stops resolving and every
fact lookup fails, so every claim silently degrades to NOT_ENOUGH_INFO (50%).

That is an unacceptable failure mode for a demo. This module is the fallback:
if Neo4j is unreachable for any reason, lookups continue to work from disk and
the verdicts stay correct.

data/kg_facts.json is the single source of truth. seed_kg.py pushes the same
file into Neo4j, so both backends always agree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

_FACTS: List[Dict[str, str]] | None = None
_INDEX: Dict[str, List[Dict[str, str]]] | None = None

_DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "kg_facts.json"


def _load() -> None:
    """Load facts once and build a lowercase name -> facts index."""
    global _FACTS, _INDEX
    if _FACTS is not None:
        return

    try:
        raw = json.loads(_DATA_FILE.read_text(encoding="utf-8"))
        facts = raw.get("facts", []) if isinstance(raw, dict) else raw
    except Exception:
        facts = []

    _FACTS = [
        {
            "subject": str(f.get("subject", "")).strip(),
            "relation": str(f.get("relation", "")).strip(),
            "object": str(f.get("object", "")).strip(),
        }
        for f in facts
        if f.get("subject") and f.get("object")
    ]

    # Index by BOTH endpoints so an edge is reachable from either side —
    # "SpaceX" must find "Elon Musk FOUNDED SpaceX" to refute a false founder.
    index: Dict[str, List[Dict[str, str]]] = {}
    for f in _FACTS:
        for endpoint in (f["subject"], f["object"]):
            index.setdefault(endpoint.lower(), []).append(f)
    _INDEX = index


def available() -> bool:
    _load()
    return bool(_FACTS)


def fact_count() -> int:
    _load()
    return len(_FACTS or [])


# Generic words that must never link two entities. "New Movie Named" matched
# "New Delhi" on the shared token "new", so a claim about a film retrieved the
# capital of India as evidence.
_GENERIC = {
    "new", "the", "and", "for", "movie", "film", "named", "name", "show",
    "song", "album", "series", "news", "video", "people", "world", "national",
    "international", "united", "general", "public", "state", "states",
    "city", "country", "party", "group", "company", "corporation",
}


def _sig_tokens(text: str) -> set:
    """
    Distinctive words only. Four-character minimum plus a generic-word block:
    a shared token has to actually identify the same entity, not merely be a
    common English word both names happen to contain.
    """
    return {
        w
        for w in str(text).lower().replace("-", " ").split()
        if len(w) >= 4 and w not in _GENERIC
    }


def query_entity_facts(entity_name: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Same contract as the Neo4j version: [{subject, relation, object}]."""
    _load()
    if not _INDEX:
        return []

    name = (entity_name or "").strip().lower()
    if len(name) < 3:
        return []

    hits: List[Dict[str, str]] = list(_INDEX.get(name, []))

    # Token-overlap fallback — NER often returns a partial or lowercased span.
    #
    # This deliberately does NOT use raw substring matching. `"in" in "xi jinping"`
    # is True, so a 2-char NER fragment used to drag in facts about Xi Jinping and
    # Donald Trump when verifying a claim about Narendra Modi, producing confident
    # nonsense verdicts. Matching whole words of 3+ chars removes that entirely.
    if not hits:
        name_tokens = _sig_tokens(name)
        if name_tokens:
            for key, facts in _INDEX.items():
                if name_tokens & _sig_tokens(key):
                    hits.extend(facts)

    seen = set()
    unique = []
    for f in hits:
        key = (f["subject"], f["relation"], f["object"])
        if key not in seen:
            seen.add(key)
            unique.append(dict(f))
    return unique[:limit]


def healthcheck() -> Dict[str, Any]:
    _load()
    return {
        "connected": bool(_FACTS),
        "backend": "local-json",
        "path": str(_DATA_FILE),
        "entities": len({f["subject"] for f in _FACTS or []}
                        | {f["object"] for f in _FACTS or []}),
        "facts": len(_FACTS or []),
        "sample": [dict(f) for f in (_FACTS or [])[:3]],
    }
