import numpy as np


def weighted_fusion(
    text_score: float,
    visual_score: float,
    source_score: float = 0.5,
    w_text: float = 0.5,
    w_visual: float = 0.4,
    w_source: float = 0.1,
) -> float:
    """Simple weighted average. visual_score is visual_fake_prob (inverted internally)."""
    text_score = float(np.clip(text_score, 0, 1))
    visual_cred = float(np.clip(1.0 - visual_score, 0, 1))
    source_score = float(np.clip(source_score, 0, 1))
    return float(np.clip(
        w_text * text_score + w_visual * visual_cred + w_source * source_score,
        0, 1,
    ))


def adaptive_fusion(
    text_score: float,
    visual_fake_prob: float,
    source_score: float = 0.5,
    text_confidence: float = 0.5,
    visual_confidence: float = 0.5,
) -> dict:
    """
    Confidence-weighted adaptive fusion.

    When a modality has low confidence (no KG facts, no faces detected),
    its weight is reduced and redistributed to the other signals.
    """
    visual_cred = float(np.clip(1.0 - visual_fake_prob, 0, 1))
    text_score = float(np.clip(text_score, 0, 1))
    source_score = float(np.clip(source_score, 0, 1))
    text_conf = float(np.clip(text_confidence, 0, 1))
    vis_conf = float(np.clip(visual_confidence, 0, 1))

    w_text = 0.50 * text_conf
    w_vis = 0.40 * vis_conf
    w_src = 0.10

    total = w_text + w_vis + w_src
    if total < 1e-9:
        w_text = w_vis = w_src = 1.0 / 3.0
        total = 1.0

    w_text /= total
    w_vis /= total
    w_src /= total

    score = float(np.clip(
        w_text * text_score + w_vis * visual_cred + w_src * source_score,
        0, 1,
    ))

    return {
        "score": round(score, 4),
        "weights": {
            "text": round(w_text, 3),
            "visual": round(w_vis, 3),
            "source": round(w_src, 3),
        },
        "signals": {
            "text_credibility": round(text_score, 4),
            "visual_credibility": round(visual_cred, 4),
            "source_reputation": round(source_score, 4),
        },
    }
