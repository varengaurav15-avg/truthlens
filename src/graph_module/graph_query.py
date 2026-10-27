"""
Knowledge-graph lookups.

IMPORTANT — how facts are stored (see seed_kg.py):

    MERGE (a)-[r:FACT {type: $relation}]->(b)

The Neo4j *relationship type* is always the literal string "FACT".
The meaningful relation ("FOUNDED", "WAS_PRESIDENT_OF", ...) lives in the
`type` PROPERTY on that relationship.

Querying `type(r)` therefore returns "FACT" for every single row, which turns
a fact like ("Bill Gates", "FOUNDED", "Microsoft") into the nonsense sentence
"Bill Gates fact Microsoft". The NLI verifier then has nothing to reason about
and returns NOT_ENOUGH_INFO for everything — a permanent 50% score.

We read `coalesce(r.type, type(r))` so both storage styles work.

We also traverse edges in BOTH directions. To refute "Bill Gates founded
SpaceX" the verifier needs the fact "Elon Musk founded SpaceX", but that edge
points Elon Musk -> SpaceX. A directed query starting at "SpaceX" finds
nothing, so the refuting evidence would never be retrieved.
"""

from typing import Any, Dict, List

# Exact name match, both directions, real relation name, correct subject/object
_CYPHER_EXACT = """
MATCH (a:Entity {name: $name})-[r]-(b:Entity)
RETURN
  CASE WHEN startNode(r) = a THEN a.name ELSE b.name END AS subject,
  coalesce(r.type, type(r))                              AS relation,
  CASE WHEN startNode(r) = a THEN b.name ELSE a.name END AS object
LIMIT $lim
"""

# Fallback: case-insensitive / partial. NER often returns "bill gates" or a
# truncated span, which would miss an exact match.
_CYPHER_FUZZY = """
MATCH (a:Entity)-[r]-(b:Entity)
WHERE toLower(a.name) = toLower($name)
   OR toLower(a.name) CONTAINS toLower($name)
   OR toLower($name)  CONTAINS toLower(a.name)
RETURN
  CASE WHEN startNode(r) = a THEN a.name ELSE b.name END AS subject,
  coalesce(r.type, type(r))                              AS relation,
  CASE WHEN startNode(r) = a THEN b.name ELSE a.name END AS object
LIMIT $lim
"""


def _driver():
    from neo4j import GraphDatabase

    from src.config.settings import settings

    return GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )


def _run(session, cypher: str, name: str, limit: int) -> List[Dict[str, Any]]:
    result = session.run(cypher, name=name, lim=limit)
    return [
        {
            "subject": rec["subject"],
            "relation": rec["relation"],
            "object": rec["object"],
        }
        for rec in result
        if rec["subject"] and rec["object"]
    ]


def _query_neo4j(name: str, limit: int) -> List[Dict[str, Any]]:
    """Raise on failure so the caller can fall back to the local store."""
    driver = _driver()
    try:
        with driver.session() as session:
            facts = _run(session, _CYPHER_EXACT, name, limit)
            if not facts:
                facts = _run(session, _CYPHER_FUZZY, name, limit)
        return facts
    finally:
        try:
            driver.close()
        except Exception:
            pass


def query_entity_facts(entity_name: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Return facts connected to `entity_name`, in either direction.

    Tries Neo4j first; falls back to the bundled offline store if the database
    is unreachable (Aura free instances are deleted after 30 days idle) or has
    no matching rows. Each dict is {subject, relation, object}, ordered the way
    the edge actually points so the sentence reads correctly.
    """
    # 3-char minimum: shorter NER fragments ("in", "US", "of") match far too
    # broadly under CONTAINS and pull in facts about unrelated entities.
    name = (entity_name or "").strip()
    if not name or len(name) < 3:
        return []

    facts: List[Dict[str, Any]] = []
    try:
        facts = _query_neo4j(name, limit)
    except Exception:
        facts = []

    if not facts:
        try:
            from . import local_store

            facts = local_store.query_entity_facts(name, limit)
        except Exception:
            facts = []

    seen = set()
    unique = []
    for f in facts:
        key = (f.get("subject"), f.get("relation"), f.get("object"))
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique[:limit]


def kg_healthcheck() -> Dict[str, Any]:
    """Report which backend is serving facts, and how many it holds."""
    driver = None
    try:
        driver = _driver()
        with driver.session() as session:
            nodes = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            edges = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            rows = [
                dict(s)
                for s in session.run(
                    "MATCH (a:Entity)-[r]->(b:Entity) "
                    "RETURN a.name AS subject, coalesce(r.type, type(r)) AS relation, "
                    "b.name AS object LIMIT 3"
                )
            ]
        if edges:
            return {
                "connected": True,
                "backend": "neo4j",
                "entities": nodes,
                "facts": edges,
                "sample": rows,
            }
        neo_error = "connected but empty — run seed_kg.py"
    except Exception as exc:
        neo_error = str(exc)
    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception:
                pass

    from . import local_store

    hc = local_store.healthcheck()
    hc["neo4j_error"] = neo_error
    return hc
