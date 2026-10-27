from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from PIL import Image
import io

from src.api.routes_vision import _analyze_image_array
from src.fusion_module.scoring import compute_final_score
from src.text_module.model import run_text_pipeline

router = APIRouter()


@router.post("/analyze")
async def analyze_multimodal(
    text: str = Form(""),
    source_reputation: float = Form(0.5),
    file: Optional[UploadFile] = File(None),
):
    """
    Full multimodal credibility analysis (text + image).

    Returns fused credibility_score, verdict, risk_level, confidence,
    per-modality signal breakdown, and Grad-CAM heatmap (base64 PNG).
    """
    if not text.strip() and file is None:
        raise HTTPException(status_code=400, detail="Provide text and/or an image.")

    # Text pipeline
    text_result = {}
    text_score = 0.5
    text_confidence = 0.1
    text_explanation = "No text provided."

    if text.strip():
        try:
            # Blocking: model inference plus Wikipedia and LLM network calls.
            # Must not run inline in an async handler or it stalls the loop
            # for every other user on the site.
            text_result = await run_in_threadpool(run_text_pipeline, text)
            text_score = text_result["credibility_score"]
            n = max(
                text_result.get("supported_count", 0)
                + text_result.get("refuted_count", 0)
                + text_result.get("nei_count", 0),
                1,
            )
            verified = (
                text_result.get("supported_count", 0)
                + text_result.get("refuted_count", 0)
            )
            text_confidence = min(0.9, 0.2 + 0.8 * (verified / n))
            text_explanation = text_result.get("explanation", "")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Text analysis failed: {exc}")

    # Vision pipeline
    vision_result = {}
    visual_fake_prob = 0.0
    visual_confidence = 0.1
    visual_explanation = "No image provided."

    if file:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        try:
            raw = await file.read()
            if len(raw) > 12 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Image too large. Limit is 12 MB.")
            img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
            vision_result = await run_in_threadpool(_analyze_image_array, img)
            visual_fake_prob = vision_result["is_fake_prob"]
            visual_explanation = vision_result.get("explanation", "")
            num_faces = vision_result.get("num_faces", 0)
            visual_confidence = 0.4 + 0.5 * min(num_faces, 3) / 3.0
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Vision analysis failed: {exc}")

    # Fusion
    fusion = compute_final_score(
        text_credibility=text_score,
        visual_fake_prob=visual_fake_prob,
        source_reputation=source_reputation,
        text_confidence=text_confidence,
        visual_confidence=visual_confidence,
    )

    return {
        **fusion,
        "text_analysis": {
            "credibility_score": text_score,
            "confidence": round(text_confidence, 3),
            "explanation": text_explanation,
            "claims": text_result.get("claims", []),
            "supported_count": text_result.get("supported_count", 0),
            "refuted_count": text_result.get("refuted_count", 0),
            "nei_count": text_result.get("nei_count", 0),
        },
        "vision_analysis": {
            "credibility_score": vision_result.get("credibility_score", 1.0),
            "is_fake_prob": visual_fake_prob,
            "cnn_fake_prob": vision_result.get("cnn_fake_prob", 0.0),
            "frequency_fake_score": vision_result.get("frequency_fake_score", 0.0),
            "high_frequency_energy": vision_result.get("high_frequency_energy", 0.0),
            "num_faces": vision_result.get("num_faces", 0),
            "confidence": round(visual_confidence, 3),
            "explanation": visual_explanation,
            "gradcam_heatmap_b64": vision_result.get("gradcam_heatmap_b64"),
        },
    }
