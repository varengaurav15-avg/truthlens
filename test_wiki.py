"""
Wikipedia retrieval diagnostic.

    .\\venv\\Scripts\\python.exe test_wiki.py

Every eval.py failure reported "no evidence provided", which means
gather_evidence() returned an empty list. The pipeline swallows exceptions
there so a broken network never crashes a scan — good for production, useless
for debugging. This script runs the same path with NOTHING suppressed.
"""

import sys
import traceback

G, R, Y, D, X = "\033[92m", "\033[91m", "\033[93m", "\033[90m", "\033[0m"


def step(n, title):
    print(f"\n{'=' * 70}\n{n}. {title}\n{'=' * 70}")


ok = True

# ---------------------------------------------------------------- 1. settings
step(1, "SETTINGS IMPORT")
try:
    from src.config.settings import settings

    print(f"{G}loaded{X}")
    print(f"  enable_wiki_evidence = {getattr(settings, 'enable_wiki_evidence', 'MISSING')}")
    print(f"  wiki_timeout         = {getattr(settings, 'wiki_timeout', 'MISSING')}")
    print(f"  llm_provider         = {getattr(settings, 'llm_provider', 'MISSING')}")
    print(f"  llm_api_key set      = {bool(str(getattr(settings, 'llm_api_key', '')).strip())}")
except Exception:
    ok = False
    print(f"{R}FAILED — this alone disables Wikipedia retrieval entirely{X}")
    traceback.print_exc()
    print(f"\n{Y}A stray key in .env that Settings doesn't declare will do this.{X}")

# ---------------------------------------------------------------- 2. requests
step(2, "HTTP CLIENT")
try:
    import requests

    print(f"{G}requests {requests.__version__}{X}")
except ImportError:
    print(f"{Y}requests missing — falling back to urllib{X}")

# ---------------------------------------------------------------- 3. network
step(3, "WIKIPEDIA API REACHABLE")
try:
    from src.graph_module.wiki_source import API, UA, _get_json

    data = _get_json(
        {"action": "query", "list": "search", "srsearch": "Barack Obama", "srlimit": 1},
        timeout=10.0,
    )
    hits = data.get("query", {}).get("search", [])
    if hits:
        print(f"{G}reachable{X} — top hit: {hits[0]['title']}")
    else:
        ok = False
        print(f"{R}responded but returned no results{X}\n{data}")
except Exception:
    ok = False
    print(f"{R}UNREACHABLE{X}")
    traceback.print_exc()
    print(f"\n{Y}Corporate proxy, firewall or offline. Wikipedia evidence cannot work.{X}")

# ---------------------------------------------------------------- 4. titles
step(4, "ENTITY -> ARTICLE RESOLUTION")
try:
    from src.graph_module.wiki_source import search_titles

    for q in ["Amitabh Bachchan", "Taylor Swift", "Barrack Obamma", "Narendra Modi"]:
        t = search_titles(q, limit=2, timeout=10.0)
        col = G if t else R
        print(f"  {q:<20} -> {col}{t}{X}")
except Exception:
    ok = False
    traceback.print_exc()

# ---------------------------------------------------------------- 5. sentences
step(5, "ARTICLE -> SENTENCES")
try:
    from src.graph_module.wiki_source import page_sentences

    sents = page_sentences("Amitabh Bachchan", timeout=12.0)
    print(f"  sentences kept: {len(sents)}")
    for s in sents[:3]:
        print(f"    [{s['section']:<10}] {s['text'][:80]}")
    if not sents:
        ok = False
        print(f"  {R}none — section filter or parser is dropping everything{X}")
except Exception:
    ok = False
    traceback.print_exc()

# ---------------------------------------------------------------- 6. evidence
step(6, "FULL EVIDENCE GATHERING (the six failing claims)")
try:
    from src.graph_module.wiki_source import gather_evidence
    from src.text_module.ner import get_entities

    CLAIMS = [
        "Narendra Modi is the Prime Minister of Canada.",
        "Barrack Obamma was the Presidnet of the USA.",
        "Amitabh Bachchan is a retired defence lawyer.",
        "Amitabh Bachchan was found dead.",
        "Robert Downey Jr is a billionaire inventor named Tony Stark.",
        "Taylor Swift won the 2019 Nobel Prize in Physics.",
    ]
    for c in CLAIMS:
        try:
            ents = [e["text"] for e in get_entities(c)]
        except Exception as exc:
            ents = []
            print(f"  {R}NER failed: {exc}{X}")
        ev = gather_evidence(c, ents, timeout=12.0)
        col = G if ev else R
        print(f"\n  {c}")
        print(f"    entities : {ents}")
        print(f"    evidence : {col}{len(ev)} sentence(s){X}")
        for e in ev[:2]:
            print(f"      {D}[{e['relevance']:.2f}] ({e['source']}) {e['text'][:78]}{X}")
        if not ev:
            ok = False
except Exception:
    ok = False
    traceback.print_exc()

print(f"\n{'=' * 70}")
print(f" {G}Retrieval healthy.{X}" if ok else f" {R}Retrieval broken — see the first FAILED step above.{X}")
print(f"{'=' * 70}\n")
sys.exit(0 if ok else 1)
