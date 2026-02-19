"""Diabetic retinopathy classifier — ViT fine-tuned on fundus images for MediVan AI."""
import random
import logging
from PIL import Image
from backend.config import MOCK_MODE, EYE_MODEL

logger = logging.getLogger(__name__)

# APTOS/EyePACS DR grading scale
CLASSES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

RISK_MAP = {
    "No DR": "low",
    "Mild": "low",
    "Moderate": "moderate",
    "Severe": "high",
    "Proliferative": "high",
}

# Map common model output labels to DR grades
LABEL_VARIANTS = {
    "no dr": "No DR", "no_dr": "No DR", "0": "No DR", "no diabetic retinopathy": "No DR",
    "healthy": "No DR", "normal": "No DR", "no retinopathy": "No DR",
    "mild": "Mild", "1": "Mild", "mild npdr": "Mild", "mild non-proliferative": "Mild",
    "moderate": "Moderate", "2": "Moderate", "moderate npdr": "Moderate",
    "moderate non-proliferative": "Moderate",
    "severe": "Severe", "3": "Severe", "severe npdr": "Severe",
    "severe non-proliferative": "Severe",
    "proliferative": "Proliferative", "4": "Proliferative", "pdr": "Proliferative",
    "proliferative dr": "Proliferative", "proliferative_dr": "Proliferative",
}

_model = None
_processor = None
_device = "cpu"


def _load():
    """Load the pretrained diabetic retinopathy ViT model."""
    global _model, _processor, _device
    if _model is not None:
        return

    import torch
    from transformers import AutoModelForImageClassification, AutoImageProcessor

    if torch.cuda.is_available():
        _device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"

    logger.info(f"Loading DR classifier '{EYE_MODEL}' on {_device}")
    _processor = AutoImageProcessor.from_pretrained(EYE_MODEL)
    _model = AutoModelForImageClassification.from_pretrained(EYE_MODEL)
    _model = _model.to(_device).eval()

    id2label = _model.config.id2label or {}
    logger.info(f"DR model labels: {id2label}")
    logger.info(f"DR classifier loaded ({sum(p.numel() for p in _model.parameters())/1e6:.1f}M params)")


def _normalize_label(label: str) -> str:
    """Normalize model output label to DR grade."""
    label_lower = label.lower().strip()
    if label_lower in LABEL_VARIANTS:
        return LABEL_VARIANTS[label_lower]
    for key, val in LABEL_VARIANTS.items():
        if key in label_lower:
            return val
    # Try numeric
    try:
        idx = int(label_lower)
        if 0 <= idx < len(CLASSES):
            return CLASSES[idx]
    except ValueError:
        pass
    return label


def classify(image: Image.Image) -> dict:
    """Classify a fundus image for diabetic retinopathy. Returns DR grade, confidence, risk."""
    if MOCK_MODE:
        return _mock()

    _load()
    import torch

    try:
        # Some DR models expect specific preprocessing (e.g., center crop, green channel)
        # AutoImageProcessor handles model-specific preprocessing
        inputs = _processor(images=image, return_tensors="pt")
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]

        id2label = _model.config.id2label or {i: str(i) for i in range(len(probs))}

        raw_results = {}
        canonical_results = {}
        for i in range(len(probs)):
            raw_label = id2label.get(i, str(i))
            norm_label = _normalize_label(raw_label)
            score = float(probs[i])
            raw_results[raw_label] = score
            canonical_results[norm_label] = canonical_results.get(norm_label, 0.0) + score

        best = max(canonical_results, key=canonical_results.get)
        conf = canonical_results[best]
        risk = RISK_MAP.get(best, "moderate")

        # Calculate composite severity score (0-4 weighted)
        severity_score = 0
        for grade, score in canonical_results.items():
            if grade in CLASSES:
                severity_score += CLASSES.index(grade) * score

        return {
            "classification": best,
            "confidence": round(conf, 4),
            "risk_level": risk,
            "dr_grade": CLASSES.index(best) if best in CLASSES else -1,
            "severity_score": round(severity_score, 2),
            "all_scores": {k: round(v, 4) for k, v in sorted(canonical_results.items(), key=lambda x: -x[1])},
            "raw_model_output": {k: round(v, 4) for k, v in sorted(raw_results.items(), key=lambda x: -x[1])[:5]},
            "recommendation": _recommendation(best),
        }

    except Exception as e:
        logger.error(f"DR classification failed: {e}", exc_info=True)
        return {
            "classification": "error",
            "confidence": 0,
            "risk_level": "moderate",
            "all_scores": {},
            "recommendation": f"Classification failed: {e}. Please re-upload or refer to ophthalmologist.",
            "error": str(e),
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
        "dr_grade": CLASSES.index(best),
        "severity_score": round(CLASSES.index(best) * conf, 2),
        "all_scores": dict(sorted(scores.items(), key=lambda x: -x[1])),
        "recommendation": _recommendation(best),
    }


def _recommendation(grade: str) -> str:
    recs = {
        "No DR": "No diabetic retinopathy detected. Continue annual dilated eye exam. Maintain HbA1c < 7% and blood pressure < 130/80.",
        "Mild": "Mild nonproliferative DR (microaneurysms only). Re-screen in 9-12 months. Optimize glycemic control — every 1% HbA1c reduction cuts DR risk by 35%.",
        "Moderate": "Moderate NPDR detected (microaneurysms + hemorrhages/exudates). Refer to ophthalmologist within 3-6 months. Tighten glucose (HbA1c < 7%) and BP (< 130/80) control urgently.",
        "Severe": "Severe NPDR detected (extensive hemorrhages, venous beading, IRMA). URGENT ophthalmology referral within 2-4 weeks — 50% risk of progression to PDR within 1 year.",
        "Proliferative": "URGENT: Proliferative DR detected (neovascularization). Immediate ophthalmology referral for anti-VEGF injection or panretinal photocoagulation. Risk of vitreous hemorrhage and retinal detachment.",
    }
    return recs.get(grade, "Refer to ophthalmologist for comprehensive dilated eye examination.")


def get_status() -> dict:
    if MOCK_MODE:
        return {"name": "DR Classifier", "status": "ready (mock)", "model": EYE_MODEL}
    return {
        "name": "DR Classifier",
        "status": "loaded" if _model else "not_loaded",
        "model": EYE_MODEL,
        "device": _device if _model else None,
    }
