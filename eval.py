"""
Automated accuracy evaluation.

    .\\venv\\Scripts\\python.exe eval.py            # full suite
    .\\venv\\Scripts\\python.exe eval.py --quick    # 12 cases, faster

Runs a labelled set of claims through the real pipeline and reports accuracy
per category, so you don't have to test by hand. Every case below is one that
has actually broken at some point in this project.

Scoring note: NOT_ENOUGH_INFO on a claim the graph genuinely can't reach is
counted as CORRECT when that is the labelled answer. A fact-checker admitting
ignorance is right behaviour, not a miss.
"""

import argparse
import sys
import time
from collections import defaultdict

G, R, Y, D, B, X = (
    "\033[92m", "\033[91m", "\033[93m", "\033[90m", "\033[94m", "\033[0m",
)

# (claim, expected_label, category)
CASES = [
    # --- straightforward true -------------------------------------------
    ("Barack Obama was President of the United States.", "SUPPORTED", "basic true"),
    ("Elon Musk founded SpaceX.", "SUPPORTED", "basic true"),
    ("Bill Gates founded Microsoft.", "SUPPORTED", "basic true"),
    ("The Moon orbits the Earth.", "SUPPORTED", "basic true"),
    ("Narendra Modi is the Prime Minister of India.", "SUPPORTED", "basic true"),

    # --- straightforward false ------------------------------------------
    ("Bill Gates founded SpaceX.", "REFUTED", "basic false"),
    ("Barack Obama was never President of the United States.", "REFUTED", "negation"),
    ("Narendra Modi is the Prime Minister of Canada.", "REFUTED", "basic false"),
    ("COVID-19 is caused by 5G radiation.", "REFUTED", "basic false"),
    ("The Earth orbits the Moon.", "REFUTED", "reversed relation"),

    # --- spelling and casing --------------------------------------------
    ("barack obama was the president of the united states", "SUPPORTED", "lowercase"),
    ("Barrack Obamma was the Presidnet of the USA.", "SUPPORTED", "typos"),
    ("bill gates founded microsft", "SUPPORTED", "typos + lowercase"),
    ("elon musk founded spacex", "SUPPORTED", "lowercase"),

    # --- fragments and informal input ------------------------------------
    ("osama the president of usa", "REFUTED", "fragment"),
    ("modi pm of india", "SUPPORTED", "fragment + acronym"),

    # --- closed-world / absence reasoning --------------------------------
    ("Osama bin Laden was the President of the United States.", "REFUTED", "closed world"),

    # --- fiction must not be read as biography ---------------------------
    ("Amitabh Bachchan is a retired defence lawyer.", "REFUTED", "fiction vs fact"),
    ("Amitabh Bachchan was found dead.", "REFUTED", "death hoax"),
    ("Robert Downey Jr is a billionaire inventor named Tony Stark.", "REFUTED", "fiction vs fact"),

    # --- genuinely unverifiable ------------------------------------------
    ("Taylor Swift won the 2019 Nobel Prize in Physics.", "REFUTED", "false award"),
    ("My neighbour Rajesh bought a red car yesterday.", "NOT_ENOUGH_INFO", "unverifiable"),
    ("The meeting has been rescheduled to Thursday.", "NOT_ENOUGH_INFO", "unverifiable"),
]

QUICK = [0, 1, 5, 6, 7, 10, 11, 14, 16, 18, 21, 22]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="run a 12-case subset")
    ap.add_argument("--verbose", action="store_true", help="show evidence for every case")
    args = ap.parse_args()

    cases = [CASES[i] for i in QUICK] if args.quick else CASES

    print(f"\n{'=' * 74}\n TRUTHLENS ACCURACY EVALUATION\n{'=' * 74}")

    # --- environment report ---------------------------------------------
    try:
        from src.graph_module.graph_query import kg_healthcheck
        from src.text_module.llm_judge import status as judge_status

        kg = kg_healthcheck()
        judge = judge_status()
        print(f"  knowledge graph : {kg.get('backend')} ({kg.get('facts', 0)} facts)")
        print(f"  adjudicator     : {judge['backend']}"
              + (f" / {judge['model']}" if judge.get("model") else ""))
        if not judge["enabled"]:
            print(f"  {Y}no LLM configured — NLI only. Expect noticeably lower accuracy.{X}")
            print(f"  {D}set llm_provider=groq and llm_api_key=... in .env{X}")
    except Exception as exc:
        print(f"  {R}environment check failed: {exc}{X}")
        return 1

    try:
        from src.text_module.model import run_text_pipeline
    except Exception as exc:
        print(f"\n{R}Cannot import pipeline: {exc}{X}")
        return 1

    print(f"\n  running {len(cases)} cases...\n{'-' * 74}")

    by_cat = defaultdict(lambda: [0, 0])
    failures = []
    t_start = time.time()

    for i, (claim, expected, category) in enumerate(cases, 1):
        t0 = time.time()
        try:
            out = run_text_pipeline(claim, max_claims=1)
            claims = out.get("claims", [])
            v = claims[0]["verification"] if claims else {}
            got = v.get("label", "NOT_ENOUGH_INFO")
        except Exception as exc:
            got, v = f"ERROR: {exc}", {}
        dt = time.time() - t0

        ok = got == expected
        by_cat[category][0] += ok
        by_cat[category][1] += 1
        mark = f"{G}PASS{X}" if ok else f"{R}FAIL{X}"

        print(f"  [{mark}] {i:>2}. {claim[:52]:<52} {dt:>5.1f}s")
        if not ok or args.verbose:
            print(f"          expected {expected} -> got {got}")
            if v.get("reason"):
                print(f"          {D}{v['reason'][:110]}{X}")
            if v.get("best_fact"):
                print(f"          {D}evidence: {v['best_fact'][:110]}{X}")
            if not ok:
                failures.append((claim, expected, got, category))

    total = sum(c[1] for c in by_cat.values())
    passed = sum(c[0] for c in by_cat.values())
    pct = 100 * passed / max(1, total)

    print(f"\n{'=' * 74}\n BY CATEGORY\n{'=' * 74}")
    for cat in sorted(by_cat):
        ok, n = by_cat[cat]
        bar = "#" * int(12 * ok / n) + "." * (12 - int(12 * ok / n))
        col = G if ok == n else (Y if ok else R)
        print(f"  {cat:<22} {col}{bar}{X}  {ok}/{n}")

    col = G if pct >= 85 else (Y if pct >= 65 else R)
    print(f"\n{'=' * 74}")
    print(f" OVERALL: {col}{passed}/{total}  ({pct:.0f}%){X}"
          f"   elapsed {time.time() - t_start:.0f}s")
    print(f"{'=' * 74}")

    if pct >= 85:
        print(f" {G}Strong. This is competitive with published fact-checking systems.{X}")
    elif pct >= 65:
        print(f" {Y}Workable. A larger adjudicator model is the biggest remaining lever.{X}")
    else:
        print(f" {R}Weak. Check that an LLM adjudicator is configured.{X}")

    if failures:
        print(f"\n FAILURES ({len(failures)}):")
        for claim, exp, got, cat in failures:
            print(f"   [{cat}] {claim[:56]}")
            print(f"        wanted {exp}, got {got}")

    print()
    return 0 if pct >= 65 else 1


if __name__ == "__main__":
    sys.exit(main())
