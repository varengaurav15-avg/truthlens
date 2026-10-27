"""
Independent benchmark against FEVER.

    .\\venv\\Scripts\\python.exe benchmark.py --per-class 50      # ~150 claims
    .\\venv\\Scripts\\python.exe benchmark.py --per-class 200     # ~600 claims

WHY THIS EXISTS
---------------
eval.py is a REGRESSION suite. I wrote those 23 cases and I wrote the code they
test, so a high score there proves only that fixes didn't break anything. It is
not evidence the system works on claims nobody designed for it.

FEVER is the standard academic benchmark for exactly this task: 185,445 claims
written by human annotators and verified against Wikipedia, labelled
SUPPORTS / REFUTES / NOT ENOUGH INFO. Published systems score roughly 70-80%
label accuracy. That is the number to compare against — not 96% on a suite
written by the same person who wrote the code.

GETTING THE DATA
----------------
Download the FEVER shared-task dev set (a .jsonl file, one claim per line) from
fever.ai, then point this script at it:

    python benchmark.py --input path\\to\\shared_task_dev.jsonl

Each line looks like:
    {"id": 1, "label": "SUPPORTS", "claim": "Roman Atwood is a content creator."}

METRICS REPORTED
----------------
Accuracy alone is misleading for a consumer fact-checker. This reports:

  * per-class precision / recall / F1
  * confusion matrix
  * FALSE-FAKE RATE — how often a TRUE claim is labelled fake. For a public
    site making assertions about real people, this is the number that carries
    reputational and legal risk. Optimise this before headline accuracy.
  * abstention rate — how often it declines to judge
  * calibration — when it says 90% confident, is it right 90% of the time?
"""

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict

G, R, Y, D, B, X = (
    "\033[92m", "\033[91m", "\033[93m", "\033[90m", "\033[94m", "\033[0m",
)

LABEL_MAP = {
    "SUPPORTS": "SUPPORTED",
    "REFUTES": "REFUTED",
    "NOT ENOUGH INFO": "NOT_ENOUGH_INFO",
    "NOTENOUGHINFO": "NOT_ENOUGH_INFO",
}
CLASSES = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]


def load_fever(path, per_class, seed=13):
    """Stratified sample so every class is equally represented."""
    buckets = defaultdict(list)
    bad = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                bad += 1
                continue
            gold = LABEL_MAP.get(str(row.get("label", "")).upper().strip())
            claim = str(row.get("claim", "")).strip()
            if gold and claim:
                buckets[gold].append(claim)

    if bad:
        print(f"  {Y}skipped {bad} unparseable lines{X}")

    rng = random.Random(seed)
    sample = []
    for cls in CLASSES:
        pool = buckets.get(cls, [])
        if not pool:
            print(f"  {Y}warning: no examples for {cls}{X}")
            continue
        rng.shuffle(pool)
        sample.extend((c, cls) for c in pool[:per_class])
    rng.shuffle(sample)
    return sample


def metrics(pairs):
    """pairs: list of (gold, pred). Returns per-class stats + confusion."""
    conf = Counter()
    for gold, pred in pairs:
        conf[(gold, pred)] += 1

    out = {}
    for cls in CLASSES:
        tp = conf[(cls, cls)]
        fp = sum(conf[(g, cls)] for g in CLASSES if g != cls)
        fn = sum(conf[(cls, p)] for p in CLASSES if p != cls)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[cls] = {"precision": prec, "recall": rec, "f1": f1,
                    "support": sum(conf[(cls, p)] for p in CLASSES)}
    return out, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/fever_dev.jsonl",
                    help="path to FEVER shared_task_dev.jsonl")
    ap.add_argument("--per-class", type=int, default=50,
                    help="claims sampled per class (3 classes)")
    ap.add_argument("--out", default="data/benchmark_results.json")
    args = ap.parse_args()

    print(f"\n{'=' * 76}\n FEVER BENCHMARK — independent evaluation\n{'=' * 76}")

    try:
        sample = load_fever(args.input, args.per_class)
    except FileNotFoundError:
        print(f"\n{R}Dataset not found: {args.input}{X}\n")
        print("Download the FEVER shared-task dev set (.jsonl) from fever.ai,")
        print("save it as data\\fever_dev.jsonl, then re-run.\n")
        print("Each line must look like:")
        print('  {"id":1,"label":"SUPPORTS","claim":"Roman Atwood is a content creator."}\n')
        return 2

    if not sample:
        print(f"{R}No usable claims parsed. Check the file format.{X}")
        return 2

    try:
        from src.text_module.llm_judge import status as judge_status
        from src.text_module.model import run_text_pipeline
    except Exception as exc:
        print(f"{R}Cannot import pipeline: {exc}{X}")
        return 1

    judge = judge_status()
    print(f"  adjudicator : {judge['backend']}"
          + (f" / {judge['model']}" if judge.get("model") else ""))
    print(f"  claims      : {len(sample)} ({args.per_class} per class)")
    est = len(sample) * 6 / 60
    print(f"  est. runtime: ~{est:.0f} min  {D}(2 LLM calls + retrieval per claim){X}")
    if judge.get("cloud"):
        print(f"  {Y}note: uses your cloud LLM quota{X}")
    print(f"{'-' * 76}")

    pairs, rows = [], []
    t0 = time.time()

    for i, (claim, gold) in enumerate(sample, 1):
        try:
            out = run_text_pipeline(claim, max_claims=1)
            cl = out.get("claims", [])
            v = cl[0]["verification"] if cl else {}
            pred = v.get("label", "NOT_ENOUGH_INFO")
            score = float(v.get("score", 0.5))
        except Exception as exc:
            pred, score = "NOT_ENOUGH_INFO", 0.5
            v = {"reason": f"error: {exc}"}

        pairs.append((gold, pred))
        rows.append({"claim": claim, "gold": gold, "pred": pred,
                     "score": score, "reason": v.get("reason", "")})

        ok = gold == pred
        if i % 10 == 0 or i == len(sample):
            acc = sum(g == p for g, p in pairs) / len(pairs)
            el = time.time() - t0
            eta = (el / i) * (len(sample) - i) / 60
            print(f"  {i:>4}/{len(sample)}  running accuracy {acc:5.1%}  "
                  f"{D}elapsed {el/60:.1f}m · eta {eta:.1f}m{X}")
        elif not ok:
            print(f"  {D}{i:>4}. {gold:<16} -> {pred:<16} {claim[:44]}{X}")

    # ---------------------------------------------------------------- report
    per_class, conf = metrics(pairs)
    total = len(pairs)
    correct = sum(g == p for g, p in pairs)
    acc = correct / total

    print(f"\n{'=' * 76}\n CONFUSION MATRIX  (rows = truth, columns = predicted)\n{'=' * 76}")
    hdr = "".join(f"{c[:12]:>16}" for c in CLASSES)
    print(f"  {'':<18}{hdr}")
    for g in CLASSES:
        cells = "".join(f"{conf[(g, p)]:>16}" for p in CLASSES)
        print(f"  {g:<18}{cells}")

    print(f"\n{'=' * 76}\n PER-CLASS\n{'=' * 76}")
    print(f"  {'class':<18}{'precision':>12}{'recall':>10}{'f1':>10}{'support':>10}")
    for cls in CLASSES:
        m = per_class[cls]
        print(f"  {cls:<18}{m['precision']:>12.1%}{m['recall']:>10.1%}"
              f"{m['f1']:>10.1%}{m['support']:>10}")

    # safety + behaviour metrics
    true_claims = [(g, p) for g, p in pairs if g == "SUPPORTED"]
    false_fake = sum(1 for g, p in true_claims if p == "REFUTED")
    ff_rate = false_fake / len(true_claims) if true_claims else 0.0
    abstain = sum(1 for _, p in pairs if p == "NOT_ENOUGH_INFO") / total

    print(f"\n{'=' * 76}\n HEADLINE\n{'=' * 76}")
    col = G if acc >= 0.70 else (Y if acc >= 0.55 else R)
    print(f"  label accuracy      {col}{acc:.1%}{X}   ({correct}/{total})")
    print(f"  {D}published FEVER systems: roughly 70-80%{X}")

    fcol = G if ff_rate <= 0.05 else (Y if ff_rate <= 0.12 else R)
    print(f"\n  FALSE-FAKE RATE     {fcol}{ff_rate:.1%}{X}   "
          f"({false_fake} true claims wrongly called fake)")
    print(f"  {D}this is the number that carries legal and reputational risk{X}")
    print(f"\n  abstention rate     {abstain:.1%}   {D}(answered NOT_ENOUGH_INFO){X}")

    # calibration
    print(f"\n{'=' * 76}\n CALIBRATION  (is the confidence meaningful?)\n{'=' * 76}")
    buckets = defaultdict(list)
    for (gold, pred), row in zip(pairs, rows):
        conf_pct = row["score"] if pred == "SUPPORTED" else (
            1 - row["score"] if pred == "REFUTED" else 0.5)
        buckets[min(int(conf_pct * 10) / 10, 0.9)].append(gold == pred)
    for b in sorted(buckets):
        hits = buckets[b]
        rate = sum(hits) / len(hits)
        bar = "#" * int(rate * 20)
        print(f"  confidence {b:.1f}-{b+0.1:.1f}  actual {rate:5.1%}  {bar:<20} n={len(hits)}")

    try:
        import pathlib
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(
            {"accuracy": acc, "false_fake_rate": ff_rate, "abstention": abstain,
             "per_class": per_class, "n": total,
             "adjudicator": judge, "rows": rows},
            open(args.out, "w", encoding="utf-8"), indent=2,
        )
        print(f"\n  full results written to {args.out}")
    except Exception as exc:
        print(f"  {Y}could not save results: {exc}{X}")

    print(f"\n{'=' * 76}")
    if acc >= 0.70:
        print(f" {G}Competitive with published FEVER systems.{X}")
    elif acc >= 0.55:
        print(f" {Y}Below published systems but well above chance (33%).{X}")
    else:
        print(f" {R}Near chance. Do not describe this as production-ready.{X}")
    print(f"{'=' * 76}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
