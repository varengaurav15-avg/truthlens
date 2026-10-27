"""
Seed the Neo4j knowledge graph with sample facts for demo/testing.
Run after Neo4j is started:  python seed_kg.py
"""
from neo4j import GraphDatabase
from src.config.settings import settings

FACTS = [
    # US Politics
    ("Barack Obama", "WAS_PRESIDENT_OF", "United States"),
    ("Barack Obama", "BORN_IN", "Hawaii"),
    ("Barack Obama", "SERVED_AS", "44th President"),
    ("Joe Biden", "WAS_PRESIDENT_OF", "United States"),
    ("Joe Biden", "SERVED_AS", "46th President"),
    ("Donald Trump", "WAS_PRESIDENT_OF", "United States"),
    ("Donald Trump", "SERVED_AS", "45th President"),
    ("Kamala Harris", "SERVED_AS", "Vice President"),
    ("United States", "CAPITAL_IS", "Washington DC"),
    ("United States", "HAS_CONSTITUTION", "US Constitution"),

    # India Politics
    ("Narendra Modi", "IS_PRIME_MINISTER_OF", "India"),
    ("Narendra Modi", "BELONGS_TO_PARTY", "BJP"),
    ("India", "CAPITAL_IS", "New Delhi"),
    ("India", "IS_A", "democratic republic"),
    ("Rahul Gandhi", "BELONGS_TO_PARTY", "Indian National Congress"),

    # Science / Health
    ("COVID-19", "IS_CAUSED_BY", "SARS-CoV-2 virus"),
    ("COVID-19 vaccine", "DEVELOPED_BY", "multiple pharmaceutical companies"),
    ("COVID-19 vaccine", "APPROVED_BY", "WHO"),
    ("Bill Gates", "FOUNDED", "Microsoft"),
    ("Bill Gates", "FUNDS", "Bill and Melinda Gates Foundation"),
    ("Earth", "ORBITS", "Sun"),
    ("Earth", "IS_A", "planet"),
    ("Moon", "ORBITS", "Earth"),

    # Technology
    ("Elon Musk", "FOUNDED", "SpaceX"),
    ("Elon Musk", "OWNS", "X (Twitter)"),
    ("Elon Musk", "CO_FOUNDED", "Tesla"),
    ("Apple", "FOUNDED_BY", "Steve Jobs"),
    ("Google", "FOUNDED_BY", "Larry Page and Sergey Brin"),
    ("Microsoft", "FOUNDED_BY", "Bill Gates"),
    ("Meta", "FOUNDED_BY", "Mark Zuckerberg"),

    # World leaders
    ("Vladimir Putin", "IS_PRESIDENT_OF", "Russia"),
    ("Xi Jinping", "IS_PRESIDENT_OF", "China"),
    ("Emmanuel Macron", "IS_PRESIDENT_OF", "France"),
    ("United Kingdom", "CAPITAL_IS", "London"),
    ("Russia", "CAPITAL_IS", "Moscow"),
    ("China", "CAPITAL_IS", "Beijing"),
]

CYPHER = """
MERGE (a:Entity {name: $subject})
MERGE (b:Entity {name: $object})
MERGE (a)-[r:FACT {type: $relation}]->(b)
"""


def load_facts():
    """
    data/kg_facts.json is the single source of truth — it is also what the
    offline fallback store reads, so both backends always hold identical facts.
    Falls back to the FACTS list above if the file is missing.
    """
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent / "data" / "kg_facts.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw.get("facts", []) if isinstance(raw, dict) else raw
        facts = [(f["subject"], f["relation"], f["object"]) for f in rows]
        if facts:
            print(f"Loaded {len(facts)} facts from {path.name}")
            return facts
    except Exception as exc:
        print(f"Could not read {path.name} ({exc}) — using built-in list")
    return FACTS


def seed():
    facts = load_facts()
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    with driver.session() as session:
        for subject, relation, obj in facts:
            session.run(CYPHER, subject=subject, relation=relation, object=obj)
    driver.close()
    print(f"Seeded {len(facts)} facts into Neo4j.")


if __name__ == "__main__":
    seed()
