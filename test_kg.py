"""
Knowledge-graph diagnostic.

Run this whenever a claim scores 50% and you don't know why:

    .\\venv\\Scripts\\python.exe test_kg.py

It answers three questions in order:
  1. Is Neo4j reachable and seeded?
  2. Does an entity lookup return readable facts (not "FACT")?
  3. Does the NLI verifier reach the right verdict on those facts?

A 50% score means NOT_ENOUGH_INFO. That is CORRECT when the graph holds no
relevant fact — a fact-checker should say "I don't know" rather than guess.
It is a BUG only when the fact exists but comes back malformed.
"""

import sys

from src.graph_module.graph_query import kg_healthcheck, query_entity_facts

GREEN, RED, YELLOW, DIM, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[90m", "\033[0m"


def rule(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


# ── 1. connectivity ────────────────────────────────────────────────────────
rule("1. KNOWLEDGE GRAPH BACKEND")
hc = kg_healthcheck()
backend = hc.get("backend", "unknown")

if not hc.get("connected"):
    print(f"{RED}NO BACKEND AVAILABLE{RESET}")
    print(f"  neo4j : {hc.get('neo4j_error', 'unknown')}")
    print(f"  local : data/kg_facts.json missing or empty")
    sys.exit(1)

if backend == "neo4j":
    print(f"{GREEN}Neo4j connected{RESET}")
else:
    print(f"{YELLOW}Using offline store (Neo4j unavailable){RESET}")
    print(f"  reason : {DIM}{hc.get('neo4j_error', '')}{RESET}")
    print(f"  source : {hc.get('path')}")
    print(f"  {DIM}This is a supported fallback — verdicts stay correct.{RESET}")

print(f"  entities : {hc['entities']}")
print(f"  facts    : {hc['facts']}")

if hc["facts"] == 0:
    print(f"\n{RED}Graph is empty.{RESET} Seed it:  .\\venv\\Scripts\\python.exe seed_kg.py")
    sys.exit(1)

print("\n  sample rows:")
for s in hc["sample"]:
    print(f"    {s['subject']}  --[{s['relation']}]->  {s['object']}")

if any(s["relation"] == "FACT" for s in hc["sample"]):
    print(f"\n  {RED}Relations read as literal 'FACT' — the query bug is back.{RESET}")

# ── 2. entity lookups ──────────────────────────────────────────────────────
rule("2. ENTITY LOOKUPS")
for name in ["Bill Gates", "SpaceX", "Elon Musk", "Barack Obama", "Narendra Modi"]:
    facts = query_entity_facts(name)
    status = f"{GREEN}{len(facts)} fact(s){RESET}" if facts else f"{YELLOW}none{RESET}"
    print(f"\n  {name:<16} {status}")
    for f in facts[:4]:
        print(f"      {DIM}{f['subject']} {f['relation'].replace('_', ' ').lower()} {f['object']}{RESET}")

# ── 3. end-to-end verification ─────────────────────────────────────────────
rule("3. NLI VERIFICATION (loads the model — first run is slow)")
try:
    from src.text_module.fact_verification import verify_claim_against_kg
    from src.text_module.model import _get_kg_facts_for_claim
except Exception as exc:
    print(f"{RED}Import failed: {exc}{RESET}")
    sys.exit(1)

CASES = [
    ("Bill Gates founded SpaceX in 2002.", "REFUTED"),
    ("Elon Musk founded SpaceX.", "SUPPORTED"),
    ("Bill Gates founded Microsoft.", "SUPPORTED"),
    ("Barack Obama was never President of the United States.", "REFUTED"),
    ("Narendra Modi is the Prime Minister of Canada.", "REFUTED"),
    ("Narendra Modi is the Prime Minister of India.", "SUPPORTED"),
    ("The Moon orbits the Earth.", "SUPPORTED"),
    ("Taylor Swift won the 2019 Nobel Prize.", "NOT_ENOUGH_INFO"),
]

passed = 0
for claim, expected in CASES:
    facts = _get_kg_facts_for_claim(claim)
    result = verify_claim_against_kg(claim, facts)
    got = result["label"]
    ok = got == expected
    passed += ok
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"\n  [{mark}] {claim}")
    print(f"      expected {expected}  ->  got {got}  ({result['score']:.0%})")
    print(f"      {DIM}{result['reason']}{RESET}")
    if result.get("best_fact"):
        print(f"      {DIM}matched: {result['best_fact']}{RESET}")
    if not facts:
        print(f"      {YELLOW}no KG facts retrieved — NER found no known entity{RESET}")

rule(f"RESULT: {passed}/{len(CASES)} correct")
if passed < len(CASES):
    print(
        "\nNot every miss is a bug. A claim about an entity absent from the graph\n"
        "SHOULD return NOT_ENOUGH_INFO — the graph only holds 36 seeded facts."
    )
