"""
Fetch the FEVER dev set and write it in the format benchmark.py expects.

    .\\venv\\Scripts\\python.exe get_fever.py

Output: data/fever_dev.jsonl, one claim per line:
    {"id": 1, "label": "SUPPORTS", "claim": "Roman Atwood is a content creator."}

Only id/label/claim are needed — evidence annotations are ignored, because we
measure LABEL accuracy, not retrieval quality.

NOTE ON SCHEMAS
---------------
FEVER mirrors on the Hub disagree about column names and label encoding. Some
store the claim as `claim`, others as `hypothesis` (the NLI reformulation).
Some store labels as strings, others as integer ClassLabels where the mapping
lives in the dataset features rather than the rows.

A previous version of this script assumed string labels and silently rejected
all 19,998 rows. It now INSPECTS the schema, decodes ClassLabel integers via
the dataset's own mapping, and prints what it found when something doesn't fit
— guessing twice is worse than looking once.
"""

import json
import pathlib
import sys
from collections import Counter

OUT = pathlib.Path("data/fever_dev.jsonl")

CLAIM_COLS = ("claim", "hypothesis", "premise", "sentence", "text", "claim_text")
LABEL_COLS = ("label", "gold_label", "fever_gold_label", "verifiable_label", "labels")

# A FEVER claim is one short sentence (~50 chars). FEVER *evidence* is a
# paragraph, often prefixed with the page title and a " . " separator.
# pietrolesci/nli_fever stores evidence in `hypothesis` — the opposite of the
# usual NLI convention — so picking a column by name alone fed 317-character
# evidence paragraphs to the benchmark and scored them as claims.
MAX_SANE_CLAIM_CHARS = 200

# Any label spelling -> canonical FEVER label
LABEL_ALIASES = {
    "SUPPORTS": "SUPPORTS", "SUPPORTED": "SUPPORTS", "SUPPORT": "SUPPORTS",
    "ENTAILMENT": "SUPPORTS", "ENTAILS": "SUPPORTS", "TRUE": "SUPPORTS",
    "REFUTES": "REFUTES", "REFUTED": "REFUTES", "REFUTE": "REFUTES",
    "CONTRADICTION": "REFUTES", "CONTRADICTS": "REFUTES", "FALSE": "REFUTES",
    "NOT ENOUGH INFO": "NOT ENOUGH INFO", "NOTENOUGHINFO": "NOT ENOUGH INFO",
    "NOT_ENOUGH_INFO": "NOT ENOUGH INFO", "NEUTRAL": "NOT ENOUGH INFO",
    "NEI": "NOT ENOUGH INFO", "UNVERIFIABLE": "NOT ENOUGH INFO",
}

CANDIDATES = [
    ("pietrolesci/nli_fever", None, "dev"),
    ("fever", "v1.0", "labelled_dev"),
    ("fever", "v1.0", "paper_dev"),
    ("copenlu/fever_gold_evidence", None, "validation"),
]


def canon(value) -> str | None:
    key = str(value).upper().strip().replace("_", " ")
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]
    return LABEL_ALIASES.get(key.replace(" ", ""))


def build_decoder(ds, col):
    """
    Return a function turning a raw label cell into a canonical FEVER label.

    Handles integer ClassLabel columns by using the dataset's own int->name
    mapping, which is the part the previous version missed.
    """
    feat = ds.features.get(col)
    names = getattr(feat, "names", None)
    if names:
        print(f"      label column {col!r} is a ClassLabel: {names}")

        def decode(v):
            try:
                return canon(names[int(v)])
            except Exception:
                return canon(v)

        return decode
    return canon


def pick_claim_column(ds, candidates):
    """
    Choose the claim column by MEASURING it, not by trusting its name.

    Claims are short; evidence is long. Sampling the median length of each
    candidate and taking the shortest reliably picks the claim even when a
    repo swaps `premise` and `hypothesis`.
    """
    import statistics

    sample = [ds[i] for i in range(min(200, len(ds)))]
    stats = {}
    for col in candidates:
        lengths = [len(str(r.get(col) or "")) for r in sample]
        lengths = [n for n in lengths if n > 0]
        if lengths:
            stats[col] = statistics.median(lengths)

    if not stats:
        return None, {}

    best = min(stats, key=stats.get)
    return best, stats


def harvest(ds):
    cols = set(ds.column_names)
    label_col = next((c for c in LABEL_COLS if c in cols), None)
    present = [c for c in CLAIM_COLS if c in cols]
    print(f"      columns: {sorted(cols)}")

    if not present or not label_col:
        print(f"      no usable claim/label columns")
        return []

    claim_col, stats = pick_claim_column(ds, present)
    for col, med in sorted(stats.items(), key=lambda kv: kv[1]):
        mark = "  <- claim" if col == claim_col else ""
        kind = "" if med <= MAX_SANE_CLAIM_CHARS else "  (looks like evidence)"
        print(f"      median len  {col:<12} {med:>6.0f} chars{kind}{mark}")

    if stats.get(claim_col, 0) > MAX_SANE_CLAIM_CHARS:
        print(f"      every column looks like evidence, not claims — skipping")
        return []

    print(f"      using claim={claim_col!r} label={label_col!r}")
    decode = build_decoder(ds, label_col)

    rows, rejected = [], Counter()
    for i, r in enumerate(ds):
        claim = str(r.get(claim_col) or "").strip()
        label = decode(r.get(label_col))
        if claim and label:
            rows.append({"id": r.get("id", i), "label": label, "claim": claim})
        else:
            rejected[str(r.get(label_col))[:24]] += 1

    if rejected and not rows:
        print(f"      every row rejected. unmapped label values: "
              f"{dict(list(rejected.items())[:6])}")
    return rows


def main():
    print("\n" + "=" * 70)
    print(" FEVER DEV SET DOWNLOADER")
    print("=" * 70 + "\n")

    if OUT.exists() and OUT.stat().st_size > 1000:
        print(f"  {OUT} already exists ({OUT.stat().st_size // 1024} KB)")
        print("  delete it to re-download.\n")
        return 0

    try:
        from datasets import load_dataset
    except ImportError:
        print("  install the loader first:  python -m pip install datasets\n")
        return 1

    for repo, config, split in CANDIDATES:
        tag = repo + (f"/{config}" if config else "") + f":{split}"
        print(f"  trying {tag}")
        try:
            ds = (load_dataset(repo, config, split=split) if config
                  else load_dataset(repo, split=split))
        except Exception as exc:
            print(f"      unavailable ({str(exc)[:70]})\n")
            continue

        rows = harvest(ds)
        if not rows:
            print()
            continue

        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        import statistics

        lens = [len(r["claim"]) for r in rows]
        counts = Counter(r["label"] for r in rows)
        print(f"\n  wrote {len(rows)} claims to {OUT}")
        for k, v in counts.most_common():
            print(f"      {k:<18} {v}")

        med = statistics.median(lens)
        print(f"\n  sanity check — median claim length: {med:.0f} chars "
              f"({'OK' if med <= MAX_SANE_CLAIM_CHARS else 'TOO LONG, still evidence'})")
        print("  samples:")
        for r in rows[:3]:
            print(f"      [{r['label']:<16}] {r['claim'][:88]}")
        print(f"\n  Next:  python benchmark.py --per-class 20\n")
        return 0

    print("-" * 70)
    print(""" AUTOMATIC DOWNLOAD FAILED — do it manually

  1. https://fever.ai  ->  DATASETS  ->  FEVER (2018)
  2. Download the shared task DEVELOPMENT set (.jsonl)
  3. Save as:  data\\fever_dev.jsonl

  Each line needs at least "label" and "claim":
    {"id":1,"label":"SUPPORTS","claim":"Roman Atwood is a content creator."}
""")
    return 1


if __name__ == "__main__":
    sys.exit(main())
