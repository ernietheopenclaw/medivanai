"""Skin lesion classifier — ViT on HAM10000 categories."""
import random
from PIL import Image
from backend.config import MOCK_MODE, SKIN_MODEL

CLASSES = [
    "melanocytic nevi",
    "melanoma",
    "benign keratosis",
    "basal cell carcinoma",
    "actinic keratoses",
    "vascular lesions",
    "dermatofibroma",
]

RISK_MAP = {
    "melanoma": "high",
    "basal cell carcinoma": "high",
    "actinic keratoses": "moderate",
    "melanocytic nevi": "low",
    "benign keratosis": "low",
    "vascular lesions": "low",
    "dermatofibroma": "low",
}

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is not None:
        return
    from transformers import ViTForImageClassification, ViTImageProcessor
    _processor = ViTImageProcessor.from_pretrained(SKIN_MODEL)
    _model = ViTForImageClassification.from_pretrained(SKIN_MODEL)
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
    results = {CLASSES[i] if i < len(CLASSES) else f"class_{i}": float(probs[i]) for i in range(len(probs))}
    best = max(results, key=results.get)
    return {
        "classification": best,
        "confidence": round(results[best], 4),
        "risk_level": RISK_MAP.get(best, "moderate"),
        "all_scores": {k: round(v, 4) for k, v in sorted(results.items(), key=lambda x: -x[1])[:5]},
        "recommendation": _recommendation(best),
    }


def _mock() -> dict:
    weights = [0.35, 0.12, 0.18, 0.1, 0.1, 0.08, 0.07]
    best = random.choices(CLASSES, weights=weights, k=1)[0]
    conf = round(random.uniform(0.6, 0.95), 4)
    scores = {c: round(random.uniform(0.01, 0.1), 4) for c in CLASSES}
    scores[best] = conf
    return {
        "classification": best,
        "confidence": conf,
        "risk_level": RISK_MAP[best],
        "all_scores": dict(sorted(scores.items(), key=lambda x: -x[1])),
        "recommendation": _recommendation(best),
    }


def _recommendation(cls: str) -> str:
    recs = {
        "melanoma": "URGENT: Refer to dermatologist immediately for biopsy. Document ABCDE characteristics.",
        "basal cell carcinoma": "Refer to dermatologist for excision evaluation within 2 weeks.",
        "actinic keratoses": "Schedule dermatology follow-up. Consider cryotherapy or topical treatment.",
        "melanocytic nevi": "Benign finding. Advise routine skin self-examination and annual screening.",
        "benign keratosis": "Benign finding. No immediate action required. Monitor for changes.",
        "vascular lesions": "Generally benign. Refer if symptomatic or cosmetically concerning.",
        "dermatofibroma": "Benign finding. No treatment needed unless symptomatic.",
    }
    return recs.get(cls, "Consult dermatologist for further evaluation.")


def get_status() -> dict:
    if MOCK_MODE:
        return {"name": "Skin Classifier", "status": "ready (mock)", "model": SKIN_MODEL}
    return {"name": "Skin Classifier", "status": "loaded" if _model else "not_loaded", "model": SKIN_MODEL}

