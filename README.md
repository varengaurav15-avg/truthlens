# TruthLens — Multimodal Misinformation & Deepfake Detection

A retrieval-augmented fact-verification system. Users paste a claim, a message or
a full article; the system retrieves supporting evidence from Wikipedia, verifies
each extracted claim, and returns a credibility verdict with the evidence it used
and a plain-English correction when a statement is false.

Images are checked separately for synthetic generation using a fine-tuned Vision
Transformer, and both signals are fused into a single confidence-weighted score.

---

## Why retrieval instead of a model's memory

Asking a language model "is this true?" produces confident hallucinations. Every
verdict here is grounded in retrieved text, and the adjudicator is explicitly
instructed to use only that text. When the evidence does not settle a claim, the
system returns **NOT_ENOUGH_INFO** rather than guessing — a fact-checker that
abstains is more useful than one that is confidently wrong.

---

## Architecture

```
          user input (free text, any formatting)
                        │
         ┌──────────────▼──────────────┐
         │  Query understanding (LLM)  │  typos, casing, fragments, questions
         │  → explicit checkable claims│  "modi pm of india"
         └──────────────┬──────────────┘  → "Narendra Modi is the PM of India."
                        │
         ┌──────────────▼──────────────┐
         │  Entity recognition          │  BERT NER + capitalised-span fallback
         └──────────────┬──────────────┘
                        │
         ┌──────────────▼──────────────┐
         │  Evidence retrieval          │  Wikipedia REST + curated KG triples
         │  disk-cached, rate-limited   │  lead-weighted, fiction sections filtered
         └──────────────┬──────────────┘
                        │
         ┌──────────────▼──────────────┐
         │  NLI cross-encoder           │  entailment / contradiction / neutral
         │  → LLM adjudication          │  closed-world reasoning over all evidence
         └──────────────┬──────────────┘
                        │
              verdict + cited evidence + correction
```

The vision path runs independently: EfficientNet-B0 for Grad-CAM spatial
saliency, FFT frequency-anomaly analysis, and a fine-tuned ViT deepfake
classifier. `src/fusion_module/` combines text and image signals with weights
scaled by each modality's confidence.

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Uvicorn, Pydantic |
| Claim verification | `cross-encoder/nli-deberta-v3-small`, Llama-3.3-70B via Groq |
| Entity recognition | `dslim/bert-base-NER` |
| Deepfake detection | `prithivMLmods/Deep-Fake-Detector-v2-Model` (ViT) |
| Saliency | Grad-CAM over EfficientNet-B0 |
| Evidence | Wikipedia REST API, Neo4j (optional, with offline fallback) |
| Frontend | React 19, Vite, Three.js / React Three Fiber, GSAP |

---

## Setup

```bash
# 1. Python environment and models (~2.5 GB, one time)
powershell -ExecutionPolicy Bypass -File .\setup.ps1

# 2. Credentials
cp .env.example .env
#    add a free Groq API key from console.groq.com

# 3. Run backend + frontend
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

Manual start:

```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
cd frontend-app && npm install && npm run dev
```

API docs at `/docs`. The system runs without Neo4j and without an LLM — it
degrades to a curated fact file and NLI-only verdicts rather than failing.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Model load state, KG backend, adjudicator in use |
| `POST /text/analyze` | Claim extraction, retrieval, verdict per claim |
| `POST /vision/analyze_image` | Deepfake probability, Grad-CAM, FFT anomaly |
| `POST /fusion/analyze` | Confidence-weighted multimodal verdict |

---

## Evaluation

```bash
python eval.py                      # 23-case internal regression suite
python get_fever.py                 # fetch the FEVER dev set
python benchmark.py --per-class 50  # independent benchmark
```

`eval.py` is a **regression suite**, not a benchmark — the cases were written
alongside the code, so it verifies that fixes hold rather than measuring
real-world accuracy. `benchmark.py` runs the pipeline against
[FEVER](https://fever.ai), 185k human-written claims the system has never seen,
and reports per-class precision/recall/F1, a confusion matrix, calibration, and
a **false-fake rate** — the share of true claims wrongly labelled fake, which is
the metric that matters most for a public tool making assertions about real
people.

---

## Engineering notes

Problems worth recording, since each took real debugging:

- **Cosine similarity cannot detect negation.** "Trump was never President" scores
  ~95% similar to "Trump was President" — same words, inverted meaning. Replaced
  with NLI entailment, which classifies it as a contradiction.
- **Fiction is not biography.** Retrieval pulled a film review describing an actor
  *playing* a lawyer and concluded he *was* one. Plot, cast and reception sections
  are now excluded and the lead paragraph is weighted heavily.
- **Silent exception handling hid a dead network path.** `except: pass` made a
  throttled Wikipedia request indistinguishable from a claim with no evidence.
  Failures are now logged and never cached.
- **Large articles broke the extract endpoint.** `action=query&prop=extracts`
  renders a whole article before truncating and times out on long pages. The REST
  summary endpoint is size-independent and is tried first.

---

## Limitations

- Evidence is English Wikipedia; breaking events lag by hours to days.
- Claims about entities absent from Wikipedia return NOT_ENOUGH_INFO by design.
- Grad-CAM runs on an ImageNet backbone, so it shows spatial saliency rather than
  forensic evidence of manipulation.
- No authentication or rate limiting — not yet suitable for public deployment.

## License

MIT
