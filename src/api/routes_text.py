from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from src.text_module.model import run_text_pipeline

router = APIRouter()


MAX_TEXT_CHARS = 20_000


class TextInput(BaseModel):
    text: str
    max_claims: int = 15

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must not be empty")
        # Cap input size. Without this a single request can pin the server
        # for minutes, and it is the simplest denial-of-service vector here.
        if len(v) > MAX_TEXT_CHARS:
            raise ValueError(
                f"text is too long ({len(v)} characters). Limit is {MAX_TEXT_CHARS}."
            )
        return v

    @field_validator("max_claims")
    @classmethod
    def claims_in_range(cls, v: int) -> int:
        return max(1, min(v, 30))


@router.post("/analyze")
def analyze_text(inp: TextInput):
    """
    Analyze article text for credibility.

    Returns credibility_score, verdict, per-claim NER + KG breakdown,
    and a human-readable explanation.
    """
    try:
        result = run_text_pipeline(inp.text, max_claims=inp.max_claims)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {exc}")

    score = result["credibility_score"]
    if score >= 0.70:
        verdict = "CREDIBLE"
    elif score >= 0.40:
        verdict = "SUSPICIOUS"
    else:
        verdict = "LIKELY_FAKE"

    result["verdict"] = verdict
    return result
