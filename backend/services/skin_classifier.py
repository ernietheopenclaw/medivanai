"""Skin lesion classifier — ViT fine-tuned on HAM10000 for MediVan AI."""
import random
import logging
from PIL import Image
from backend.config import MOCK_MODE, SKIN_MODEL

logger = logging.getLogger(__name__)

# HAM10000 canonical classes
CLASSES = [
    "melanocytic nevi",
    "melanoma",
    "benign keratosis",
    "basal cell carcinoma",
    "actinic keratoses",
    "vascular lesions",
    "dermatofibroma",
]

# Abbreviation mapping (HAM10000 dataset labels -> readable names)
HAM_ABBREV = {
    "nv": "melanocytic nevi",
    "mel": "melanoma",
    "bkl": "benign keratosis",
    "bcc": "basal cell carcinoma",
    "akiec": "actinic keratoses",
    "vasc": "vascular lesions",
    "df": "dermatofibroma",
}

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
_device = "cpu"


def _load():
    """Load the pretrained skin lesion ViT model."""
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

    logger.info(f"Loading skin classifier '{SKIN_MODEL}' on {_device}")
    _processor = AutoImageProcessor.from_pretrained(SKIN_MODEL)
    _model = AutoModelForImageClassification.from_pretrained(SKIN_MODEL)
    _model = _model.to(_device).eval()

    # Log the model's label mapping
    id2label = _model.config.id2label or {}
    logger.info(f"Skin model labels: {id2label}")
    logger.info(f"Skin classifier loaded ({sum(p.numel() for p in _model.parameters())/1e6:.1f}M params)")


def _normalize_label(label: str) -> str:
    """Normalize model output label to our canonical class names."""
    label_lower = label.lower().strip()

    # Direct match to HAM abbreviations
    if label_lower in HAM_ABBREV:
        return HAM_ABBREV[label_lower]

    # Check if label contains a canonical class name
    for canonical in CLASSES:
        if canonical in label_lower or label_lower in canonical:
            return canonical

    # Fuzzy matching for common variants
    VARIANTS = {
        "nevus": "melanocytic nevi", "nevi": "melanocytic nevi", "mole": "melanocytic nevi",
        "melanocytic": "melanocytic nevi",
        "melanoma": "melanoma", "malignant melanoma": "melanoma",
        "keratosis": "benign keratosis", "seborrheic": "benign keratosis", "bkl": "benign keratosis",
        "basal": "basal cell carcinoma", "bcc": "basal cell carcinoma", "carcinoma": "basal cell carcinoma",
        "actinic": "actinic keratoses", "keratoses": "actinic keratoses", "solar": "actinic keratoses",
        "vascular": "vascular lesions", "angioma": "vascular lesions", "hemangioma": "vascular lesions",
        "dermatofibroma": "dermatofibroma", "fibroma": "dermatofibroma",
    }
    for key, val in VARIANTS.items():
        if key in label_lower:
            return val

    return label  # Return as-is if no match


def classify(image: Image.Image) -> dict:
    """Classify a skin lesion image. Returns classification, confidence, risk, recommendations."""
    if MOCK_MODE:
        return _mock()

    _load()
    import torch

    try:
        # Preprocess
        inputs = _processor(images=image, return_tensors="pt")
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = _model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]

        # Map model labels to our canonical classes
        id2label = _model.config.id2label or {i: CLASSES[i] if i < len(CLASSES) else f"class_{i}" for i in range(len(probs))}

        raw_results = {}
        canonical_results = {}
        for i in range(len(probs)):
            raw_label = id2label.get(i, f"class_{i}")
            norm_label = _normalize_label(raw_label)
            score = float(probs[i])
            raw_results[raw_label] = score
            # Aggregate scores for same canonical class
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
        logger.error(f"Skin classification failed: {e}", exc_info=True)
        return {
            "classification": "error",
            "confidence": 0,
            "risk_level": "moderate",
            "all_scores": {},
            "recommendation": f"Classification failed: {e}. Please re-upload or consult dermatologist.",
            "error": str(e),
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
        "melanoma": "URGENT: Refer to dermatologist immediately for biopsy. Document ABCDE characteristics (Asymmetry, Border irregularity, Color variation, Diameter >6mm, Evolution).",
        "basal cell carcinoma": "Refer to dermatologist for excision evaluation within 2 weeks. Most common skin cancer — high cure rate with early treatment.",
        "actinic keratoses": "Pre-cancerous lesion. Schedule dermatology follow-up. Consider cryotherapy or topical 5-FU treatment. Sun protection critical.",
        "melanocytic nevi": "Benign mole. No immediate action required. Advise routine skin self-examination (monthly) and annual professional screening.",
        "benign keratosis": "Benign finding (seborrheic keratosis). No treatment needed unless cosmetically concerning. Monitor for changes.",
        "vascular lesions": "Generally benign vascular finding. Refer if symptomatic, bleeding, or cosmetically concerning.",
        "dermatofibroma": "Benign fibrous nodule. No treatment needed unless symptomatic. Reassure patient.",
    }
    return recs.get(cls, "Consult dermatologist for further evaluation.")


def get_status() -> dict:
    if MOCK_MODE:
        return {"name": "Skin Classifier", "status": "ready (mock)", "model": SKIN_MODEL}
    return {
        "name": "Skin Classifier",
        "status": "loaded" if _model else "not_loaded",
        "model": SKIN_MODEL,
        "device": _device if _model else None,
    }
