"""Chest X-ray classifier."""
import random
from PIL import Image
from backend.config import MOCK_MODE, CHEST_MODEL

CLASSES = ["normal", "cardiomegaly", "pneumonia", "pleural effusion", "atelectasis", "pneumothorax", "tuberculosis"]

RISK_MAP = {
    "normal": "low",
    "cardiomegaly": "moderate",
    "pneumonia": "high",
    "pleural effusion": "moderate",
    "atelectasis": "moderate",
    "pneumothorax": "high",
    "tuberculosis": "high",
}

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is not None:
        return
    from transformers import ViTForImageClassification, ViTImageProcessor
    _processor = ViTImageProcessor.from_pretrained(CHEST_MODEL)
    _model = ViTForImageClassification.from_pretrained(CHEST_MODEL)
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
    results = {id2label.get(i, f"class_{i}"): float(probs[i]) for i in range(len(probs))}
    best = max(results, key=results.get)
    risk = RISK_MAP.get(best.lower(), "moderate")
    return {
        "classification": best,
        "confidence": round(results[best], 4),
        "risk_level": risk,
        "all_scores": {k: round(v, 4) for k, v in sorted(results.items(), key=lambda x: -x[1])[:5]},
        "recommendation": _recommendation(best.lower()),
    }


def _mock() -> dict:
    weights = [0.3, 0.15, 0.2, 0.12, 0.1, 0.05, 0.08]
    best = random.choices(CLASSES, weights=weights, k=1)[0]
    conf = round(random.uniform(0.65, 0.94), 4)
    scores = {c: round(random.uniform(0.01, 0.08), 4) for c in CLASSES}
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
        "normal": "No acute findings. Routine follow-up as clinically indicated.",
        "cardiomegaly": "Enlarged cardiac silhouette noted. Recommend echocardiogram and cardiology referral.",
        "pneumonia": "Consolidation pattern consistent with pneumonia. Start empiric antibiotics, consider sputum culture.",
        "pleural effusion": "Fluid collection noted. Consider thoracentesis if large. Evaluate underlying cause.",
        "atelectasis": "Lung collapse pattern. Encourage incentive spirometry. Rule out obstruction.",
        "pneumothorax": "URGENT: Air in pleural space. Assess for tension pneumothorax. May require chest tube.",
        "tuberculosis": "Pattern suggestive of TB. Isolate patient. Obtain sputum AFB and start workup.",
    }
    return recs.get(cls, "Consult radiologist for further evaluation.")


def get_status() -> dict:
    if MOCK_MODE:
        return {"name": "Chest X-ray Classifier", "status": "mock", "model": CHEST_MODEL}
    return {"name": "Chest X-ray Classifier", "status": "loaded" if _model else "not_loaded", "model": CHEST_MODEL}

