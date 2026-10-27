"""
Deepfake image classifier.

------------------------------------------------------------------------------
WHY THIS FILE WAS REWRITTEN
------------------------------------------------------------------------------
The original implementation was:

    timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)

`pretrained=True` loads ImageNet weights for the BACKBONE, but `num_classes=2`
replaces the classifier with a brand-new, RANDOMLY INITIALISED head. Nothing in
this repository ever loaded a trained checkpoint (checkpoints/ and models/ are
empty; no call to load_state_dict exists anywhere).

The result: predict_fake_prob() returned a random linear projection of ImageNet
features. It produced a confident-looking number that carried no information
whatsoever about whether an image was manipulated.

We now load a genuinely fine-tuned deepfake classifier from HuggingFace:

    prithivMLmods/Deep-Fake-Detector-v2-Model
    ViT (google/vit-base-patch16-224-in21k) fine-tuned on real vs. deepfake
    faces. Reported accuracy 92.1%, F1 0.921 over 56,001 held-out images.
    Labels: "Realism" / "Deepfake". Apache-2.0. ~86M params (~340MB download).

The EfficientNet path is kept ONLY as a Grad-CAM surface and as a last-resort
fallback, and `classifier_status()` reports honestly which one is live so the
UI never implies a trained verdict when there isn't one.
"""

from typing import Optional

import torch
import torch.nn as nn

HF_MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"

_model_cache: dict = {}          # EfficientNet cache (Grad-CAM + fallback)
_hf_cache: dict = {}             # trained ViT classifier
_hf_failed = False               # don't retry a failed download every request


# ---------------------------------------------------------------------------
# Trained classifier (primary)
# ---------------------------------------------------------------------------
def get_deepfake_classifier():
    """
    Load the fine-tuned ViT deepfake classifier. Returns (model, processor,
    fake_index) or None if unavailable — callers must handle None.
    """
    global _hf_failed
    if _hf_failed:
        return None
    if "model" in _hf_cache:
        return _hf_cache["model"], _hf_cache["processor"], _hf_cache["fake_idx"]

    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID)
        model = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID)
        model.eval()

        # Resolve which output index means "fake" rather than assuming 1
        fake_idx = 1
        id2label = getattr(model.config, "id2label", {}) or {}
        for idx, label in id2label.items():
            if any(k in str(label).lower() for k in ("fake", "deep", "synth")):
                fake_idx = int(idx)
                break

        _hf_cache.update({"model": model, "processor": processor, "fake_idx": fake_idx})
        return model, processor, fake_idx

    except Exception:
        _hf_failed = True
        return None


# ---------------------------------------------------------------------------
# EfficientNet (Grad-CAM surface / fallback only)
# ---------------------------------------------------------------------------
def get_model(num_classes: int = 2, pretrained: bool = True):
    """
    Cached EfficientNet-B0.

    WARNING: its 2-class head is untrained. Use it for Grad-CAM spatial
    saliency, not for verdicts.
    """
    key = (num_classes, pretrained)
    if key not in _model_cache:
        try:
            import timm

            m = timm.create_model(
                "efficientnet_b0", pretrained=pretrained, num_classes=num_classes
            )
        except Exception:
            from torchvision.models import efficientnet_b0

            m = efficientnet_b0(weights="IMAGENET1K_V1")
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        m.eval()
        _model_cache[key] = m
    return _model_cache[key]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_fake_prob_from_image(img) -> Optional[float]:
    """
    P(deepfake) from the trained ViT. `img` is a PIL Image or HxWx3 numpy array.
    Returns None when the trained model is unavailable.
    """
    loaded = get_deepfake_classifier()
    if loaded is None:
        return None

    model, processor, fake_idx = loaded
    try:
        from PIL import Image
        import numpy as np

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = img.convert("RGB")

        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0]
        return float(probs[fake_idx].item())
    except Exception:
        return None


def predict_fake_prob(tensor: torch.Tensor, model=None, device: str = "cpu") -> float:
    """
    Legacy tensor-based entry point, kept for backwards compatibility.

    NOTE: this runs the UNTRAINED EfficientNet head and its output is not a
    meaningful deepfake signal. routes_vision.py calls
    predict_fake_prob_from_image() instead.
    """
    if model is None:
        model = get_model()
    model.eval()
    device_obj = torch.device(device)
    model = model.to(device_obj)
    tensor = tensor.to(device_obj)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.softmax(logits, dim=1)[0, 1].item()
    return float(prob)


def classifier_status() -> dict:
    """Report which classifier is actually producing verdicts."""
    if "model" in _hf_cache:
        return {
            "backend": "vit-trained",
            "model": HF_MODEL_ID,
            "trained": True,
            "note": "Fine-tuned deepfake classifier (~92% reported accuracy).",
        }
    return {
        "backend": "efficientnet-untrained",
        "model": "efficientnet_b0",
        "trained": False,
        "note": (
            "Trained classifier unavailable — the EfficientNet head is randomly "
            "initialised, so CNN output is NOT a valid deepfake signal. The "
            "frequency-domain analysis remains meaningful."
        ),
    }


def warmup(device: str = "cpu"):
    """Pre-load both models so the first request isn't slow."""
    get_deepfake_classifier()
    m = get_model()
    predict_fake_prob(torch.zeros(1, 3, 224, 224), model=m, device=device)
