import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .routes_fusion import router as fusion_router
from .routes_text import router as text_router
from .routes_vision import router as vision_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up models...")
    t0 = time.perf_counter()

    try:
        from src.vision_module.deepfake_model import warmup as vision_warmup
        vision_warmup()
        logger.info("Vision model ready.")
    except Exception as exc:
        logger.warning(f"Vision model warmup failed (non-fatal): {exc}")

    try:
        from src.text_module.fact_verification import _get_nli_model
        _get_nli_model()
        logger.info("NLI model ready.")
    except Exception as exc:
        logger.warning(f"NLI model warmup failed (non-fatal): {exc}")

    try:
        from src.text_module.ner import _get_ner_pipeline
        _get_ner_pipeline()
        logger.info("NER pipeline ready.")
    except Exception as exc:
        logger.warning(f"NER warmup failed (non-fatal): {exc}")

    elapsed = time.perf_counter() - t0
    logger.info(f"All warmup complete in {elapsed:.1f}s")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Multimodal Misinformation & Deepfake Detector",
    description=(
        "Analyzes news text (claim extraction + KG fact-checking), "
        "images (deepfake detection + Grad-CAM), and fuses signals "
        "into a credibility score with verdict and risk level."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
# NOTE: allow_credentials MUST stay False while allow_origins is ["*"].
# The CORS spec forbids the wildcard origin on credentialed requests, and
# browsers reject the preflight outright. This app sends no cookies or auth
# headers, so wildcard + no-credentials is both correct and permissive enough
# for the Vite dev server (:5173), the Streamlit app (:8501), a production
# build (:4173), and pages opened directly from disk (origin "null").
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(text_router, prefix="/text", tags=["Text Analysis"])
app.include_router(vision_router, prefix="/vision", tags=["Vision Analysis"])
app.include_router(fusion_router, prefix="/fusion", tags=["Multimodal Fusion"])


@app.get("/", tags=["Meta"])
def root():
    return {
        "status": "ok",
        "service": "Multimodal Misinformation Detector",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Meta"])
def health():
    """Detailed health check: reports which models are loaded."""
    from src.vision_module.deepfake_model import _model_cache, classifier_status
    from src.text_module.fact_verification import _nli_model
    from src.text_module.ner import _ner_pipeline

    try:
        from src.graph_module.graph_query import kg_healthcheck
        kg = kg_healthcheck()
    except Exception as exc:
        kg = {"connected": False, "error": str(exc)}

    return {
        "status": "ok",
        "models": {
            "efficientnet": "loaded" if _model_cache else "not_loaded",
            "nli_model": "loaded" if _nli_model is not None else "not_loaded",
            "ner_pipeline": "loaded" if _ner_pipeline is not None else "not_loaded",
        },
        "classifier": classifier_status(),
        "knowledge_graph": {
            "backend": kg.get("backend", "unknown"),
            "connected": kg.get("connected", False),
            "facts": kg.get("facts", 0),
        },
        "judge": _judge_status(),
    }


def _judge_status() -> dict:
    try:
        from src.text_module.llm_judge import status
        return status()
    except Exception as exc:
        return {"backend": "none", "enabled": False, "note": str(exc)}
