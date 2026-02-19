"""Chest X-ray classifier — ViT fine-tuned for thoracic pathology for MediVan AI."""
import random
import logging
from PIL import Image
from backend.config import MOCK_MODE, CHEST_MODEL

logger = logging.getLogger(__name__)

# Common CXR findings
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

# Map common model output labels to our canonical names
LABEL_VARIANTS = {
    "normal": "normal", "no finding": "normal", "no findings": "normal", "healthy": "normal",
    "cardiomegaly": "cardiomegaly", "enlarged heart": "cardiomegaly", "cardiac enlargement": "cardiomegaly",
    "pneumonia": "pneumonia", "lung opacity": "pneumonia", "consolidation": "pneumonia", "infiltrate": "pneumonia",
    "pleural effusion": "pleural effusion", "effusion": "pleural effusion", "pleural_effusion": "pleural effusion",
    "atelectasis": "atelectasis", "collapse": "atelectasis", "lung collapse": "atelectasis",
    "pneumothorax": "pneumothorax",
    "tuberculosis": "tuberculosis", "tb": "tuberculosis", "pulmonary tuberculosis": "tuberculosis",
    # Additional CheXpert/ChestX-ray14 labels
    "edema": "pleural effusion",
    "consolidation": "pneumonia",
    "mass": "pneumonia",  # fallback
    "nodule": "normal",  # small nodules often incidental
    "emphysema": "normal",  # chronic, not acute
    "fibrosis": "normal",
    "hernia": "normal",
}

_model = None
_processor = None
_device = "cpu"


def _load():
    """Load the pretrained chest X-ray ViT model."""
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

    logger.info(f"Loading chest classifier '{CHEST_MODEL}' on {_device}")
    _processor = AutoImageProcessor.from_pretrained(CHEST_MODEL)
    _model = AutoModelForImageClassification.from_pretrained(CHEST_MODEL)
    _model = _model.to(_device).eval()

    id2label = _model.config.id2label or {}
    logger.info(f"Chest model labels: {id2label}")
    logger.info(f"Chest classifier loaded ({sum(p.numel() for p in _model.parameters())/1e6:.1f}M params)")


def _normalize_label(label: str) -> str:
    """Normalize model output label to our canonical class names."""
    label_lower = label.lower().strip()
    if label_lower in LABEL_VARIANTS:
        return LABEL_VARIANTS[label_lower]
    for key, val in LABEL_VARIANTS.items():
        if key in label_lower:
            return val
    return label


def classify(image: Image.Image) -> dict:
    """Classify a chest X-ray image."""
    if MOCK_MODE:
        return _mock()

    _load()
    import torch

    try:
        inputs = _processor(images=image, return_tensors="pt")
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]

        id2label = _model.config.id2label or {i: f"class_{i}" for i in range(len(probs))}

        raw_results = {}
        canonical_results = {}
        for i in range(len(probs)):
            raw_label = id2label.get(i, f"class_{i}")
            norm_label = _normalize_label(raw_label)
            score = float(probs[i])
            raw_results[raw_label] = score
            canonical_results[norm_label] = canonical_results.get(norm_label, 0.0) + score

        best = max(canonical_results, key=canonical_results.get)
        conf = canonical_results[best]
        risk = RISK_MAP.get(best, "moderate")

        return {
            "classification": best,
            "confidence": round(conf, 4),
            "risk_level": risk,
            "all_scores": {k: round(v, 4) for k, v in sorted(canonical_results.items(), key=lambda x: -x[1])[:7]},
            "raw_model_output": {k: round(v, 4) for k, v in sorted(raw_results.items(), key=lambda x: -x[1])[:5]},
            "recommendation": _recommendation(best),
        }

    except Exception as e:
        logger.error(f"Chest classification failed: {e}", exc_info=True)
        return {
            "classification": "error",
            "confidence": 0,
            "risk_level": "moderate",
            "all_scores": {},
            "recommendation": f"Classification failed: {e}. Please re-upload or consult radiologist.",
            "error": str(e),
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
        "normal": "No acute cardiopulmonary findings. Routine follow-up as clinically indicated.",
        "cardiomegaly": "Enlarged cardiac silhouette (CTR > 0.5). Recommend echocardiogram, BNP levels, and cardiology referral. Evaluate for heart failure.",
        "pneumonia": "Consolidation/opacity pattern consistent with pneumonia. Start empiric antibiotics per guidelines. Consider sputum culture and CRP/procalcitonin.",
        "pleural effusion": "Fluid collection in pleural space. Consider thoracentesis if large or symptomatic. Evaluate for CHF, infection, malignancy.",
        "atelectasis": "Lung collapse pattern noted. Encourage incentive spirometry and deep breathing exercises. Rule out endobronchial obstruction if persistent.",
        "pneumothorax": "URGENT: Air in pleural space detected. Assess for tension pneumothorax (tracheal deviation, hypotension). May require emergent chest tube decompression.",
        "tuberculosis": "Radiographic pattern suggestive of TB (upper lobe infiltrates/cavitation). ISOLATE patient immediately. Obtain sputum AFB x3, start TB workup per CDC guidelines.",
    }
    return recs.get(cls, "Consult radiologist for further evaluation.")


def get_status() -> dict:
    if MOCK_MODE:
        return {"name": "Chest X-ray Classifier", "status": "ready (mock)", "model": CHEST_MODEL}
    return {
        "name": "Chest X-ray Classifier",
        "status": "loaded" if _model else "not_loaded",
        "model": CHEST_MODEL,
        "device": _device if _model else None,
    }
