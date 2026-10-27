from typing import Optional

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from PIL import Image
import io

from src.vision_module.deepfake_model import (
    classifier_status,
    get_model,
    predict_fake_prob,
    predict_fake_prob_from_image,
)
from src.vision_module.explainability import (
    get_efficientnet_target_layer,
    grad_cam_saliency,
    saliency_to_heatmap_b64,
)
from src.vision_module.face_detection import detect_faces
from src.vision_module.frequency_analysis import frequency_fake_score, high_frequency_energy
from src.vision_module.preprocessing import preprocess_image

router = APIRouter()


def _analyze_image_array(img: np.ndarray) -> dict:
    """Core image analysis — returns a rich result dict."""
    orig_h, orig_w = img.shape[:2]

    # Face detection
    try:
        faces = detect_faces(img)
    except Exception:
        faces = []

    # Frequency-domain fake signal (soft prior)
    freq_fake = frequency_fake_score(img)
    hfe = round(high_frequency_energy(img), 4)

    # Deepfake inference — trained ViT classifier is the real signal.
    # predict_fake_prob_from_image() returns None if the model couldn't load,
    # in which case we fall back to frequency analysis alone rather than
    # reporting a number from the untrained EfficientNet head.
    cnn_fake_prob = predict_fake_prob_from_image(img)
    status = classifier_status()  # read after the lazy load has been attempted
    trained = cnn_fake_prob is not None

    tensor = preprocess_image(img)
    model = get_model()  # EfficientNet retained for Grad-CAM spatial saliency

    if not trained:
        cnn_fake_prob = 0.0

    # Grad-CAM saliency heatmap
    gradcam_b64: Optional[str] = None
    try:
        target_layer = get_efficientnet_target_layer(model)
        saliency = grad_cam_saliency(model, tensor.clone(), target_layer=target_layer)
        gradcam_b64 = saliency_to_heatmap_b64(saliency, orig_size=(orig_h, orig_w))
    except Exception:
        pass

    # Fuse the classifier with the frequency prior.
    # Without a trained classifier the only honest signal is frequency analysis.
    if trained:
        blended_fake_prob = round(0.85 * cnn_fake_prob + 0.15 * freq_fake, 4)
    else:
        blended_fake_prob = round(freq_fake, 4)

    credibility = round(1.0 - blended_fake_prob, 4)

    if trained:
        detail = (
            f"Classifier: {cnn_fake_prob*100:.0f}% fake, "
            f"frequency anomaly: {freq_fake*100:.0f}%."
        )
    else:
        detail = (
            f"Frequency anomaly only: {freq_fake*100:.0f}%. "
            f"Trained classifier unavailable, so no CNN verdict is included."
        )

    if blended_fake_prob > 0.65:
        headline = f"High manipulation probability ({blended_fake_prob*100:.0f}%)."
    elif blended_fake_prob > 0.35:
        headline = (
            f"Moderate manipulation signals ({blended_fake_prob*100:.0f}%). "
            f"Inconclusive — manual review recommended."
        )
    else:
        headline = f"No strong manipulation detected ({blended_fake_prob*100:.0f}% fake prob)."

    explanation = f"{headline} {detail} {len(faces)} face(s) detected."

    return {
        "credibility_score": credibility,
        "is_fake_prob": blended_fake_prob,
        "cnn_fake_prob": round(cnn_fake_prob, 4),
        "frequency_fake_score": freq_fake,
        "high_frequency_energy": hfe,
        "num_faces": len(faces),
        "face_boxes": [list(f) for f in faces],
        "explanation": explanation,
        "gradcam_heatmap_b64": gradcam_b64,
        "classifier": status,          # which model actually produced the verdict
        "classifier_trained": trained,
    }


MAX_IMAGE_BYTES = 12 * 1024 * 1024   # 12 MB


@router.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze a single image for deepfake / manipulation signals.

    CONCURRENCY NOTE: _analyze_image_array runs CNN inference and other
    blocking work. Calling it directly inside an `async def` would block the
    event loop for its entire duration, serialising every request on the
    server — two users and the second waits for the first. It is dispatched to
    a worker thread instead.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    raw = await file.read()
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large ({len(raw) // 1024} KB). Limit is {MAX_IMAGE_BYTES // 1024 // 1024} MB.",
        )

    try:
        img = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {exc}")

    try:
        return await run_in_threadpool(_analyze_image_array, img)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")
