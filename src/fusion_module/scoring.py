from typing import Dict, Any

from .fusion_model import adaptive_fusion


def _verdict(score: float) -> str:
    if score >= 0.70:
        return "CREDIBLE"
    if score >= 0.40:
        return "SUSPICIOUS"
    return "LIKELY_FAKE"


def _risk_level(score: float) -> str:
    if score >= 0.70:
        return "LOW"
    if score >= 0.40:
        return "MEDIUM"
    return "HIGH"


def compute_final_score(
    text_credibility: float,
    visual_fake_prob: float,
    source_reputation: float = 0.5,
    text_confidence: float = 0.5,
    visual_confidence: float = 0.5,
) -> Dict[str, Any]:
    """
    Fuse text, visual, and source signals into a final credibility assessment.

    Returns credibility_score, verdict, risk_level, confidence,
    fusion_weights, and per-signal scores.
    """
    fusion = adaptive_fusion(
        text_score=text_credibility,
        visual_fake_prob=visual_fake_prob,
        source_score=source_reputation,
        text_confidence=text_confidence,
        visual_confidence=visual_confidence,
    )

    score = fusion["score"]
    visual_cred = 1.0 - visual_fake_prob
    signal_spread = abs(text_credibility - visual_cred)
    avg_conf = (text_confidence + visual_confidence) / 2.0
    confidence = round(avg_conf * (1.0 - signal_spread * 0.5), 4)

    return {
        "credibility_score": score,
        "verdict": _verdict(score),
        "risk_level": _risk_level(score),
        "confidence": confidence,
        "fusion_weights": fusion["weights"],
        "signals": fusion["signals"],
    }
