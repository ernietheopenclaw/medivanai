"""Diabetic retinopathy classifier."""
import random
from PIL import Image
from backend.config import MOCK_MODE, EYE_MODEL

CLASSES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
RISK_MAP = {"No DR": "low", "Mild": "low", "Moderate": "moderate", "Severe": "high", "Proliferative": "high"}

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is not None:
        return
    from transformers import ViTForImageClassification, ViTImageProcessor
    _processor = ViTImageProcessor.from_pretrained(EYE_MODEL)
    _model = ViTForImageClassification.from_pretrained(EYE_MODEL)
    _model.eval()


def classify(image: Image.Image) -> dict:
    if MOCK_MODE:
        return _mock()
    _load()
    import torch
    inputs = _processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    id2label = _model.config.id2label
    results = {id2label.get(i, CLASSES[i] if i < len(CLASSES) else f"class_{i}"): float(probs[i]) for i in range(len(probs))}
    best = max(results, key=results.get)
    risk = RISK_MAP.get(best, "moderate")
    return {
        "classification": best,
        "confidence": round(results[best], 4),
        "risk_level": risk,
        "all_scores": {k: round(v, 4) for k, v in sorted(results.items(), key=lambda x: -x[1])},
        "recommendation": _recommendation(best),
    }


def _mock() -> dict:
    weights = [0.35, 0.2, 0.2, 0.15, 0.1]
    best = random.choices(CLASSES, weights=weights, k=1)[0]
    conf = round(random.uniform(0.6, 0.93), 4)
    scores = {c: round(random.uniform(0.01, 0.1), 4) for c in CLASSES}
    scores[best] = conf
    return {
        "classification": best,
        "confidence": conf,
        "risk_level": RISK_MAP[best],
        "all_scores": dict(sorted(scores.items(), key=lambda x: -x[1])),
        "recommendation": _recommendation(best),
    }


def _recommendation(grade: str) -> str:
    recs = {
        "No DR": "No diabetic retinopathy detected. Annual eye exam recommended. Continue diabetes management.",
        "Mild": "Mild nonproliferative DR. Re-screen in 9-12 months. Optimize glycemic control (HbA1c < 7%).",
        "Moderate": "Moderate NPDR. Refer to ophthalmologist within 3-6 months. Tighten glucose and BP control.",
        "Severe": "Severe NPDR. URGENT ophthalmology referral within 2-4 weeks. High risk of progression.",
        "Proliferative": "URGENT: Proliferative DR detected. Immediate ophthalmology referral for anti-VEGF/laser treatment.",
    }
    return recs.get(grade, "Refer to ophthalmologist for evaluation.")


def get_status() -> dict:
    if MOCK_MODE:
        return {"name": "DR Classifier", "status": "mock", "model": EYE_MODEL}
    return {"name": "DR Classifier", "status": "loaded" if _model else "not_loaded", "model": EYE_MODEL}

