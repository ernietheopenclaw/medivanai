"""CLIP/SigLIP zero-shot image router for MediVan AI."""
import random
import logging
from PIL import Image
from backend.config import MOCK_MODE, CLIP_MODEL

logger = logging.getLogger(__name__)

IMAGE_TYPES = ["skin_lesion", "chest_xray", "fundus"]
PROMPTS = {
    "skin_lesion": [
        "a dermoscopic image of a skin lesion",
        "a close-up photograph of a mole or skin growth",
        "a dermatology clinical image of skin",
    ],
    "chest_xray": [
        "a chest x-ray radiograph",
        "a frontal chest radiograph showing lungs and heart",
        "a medical x-ray image of the thorax",
    ],
    "fundus": [
        "a fundus photograph of the retina",
        "an ophthalmoscopic image of the eye retina",
        "a retinal fundus image showing blood vessels and optic disc",
    ],
}

_model = None
_processor = None
_tokenizer = None
_device = "cpu"


def _load_model():
    """Load CLIP model for zero-shot image classification."""
    global _model, _processor, _tokenizer, _device
    if _model is not None:
        return

    import torch

    # Determine device
    if torch.cuda.is_available():
        _device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"
    logger.info(f"Loading CLIP model '{CLIP_MODEL}' on {_device}")

    try:
        # Try open_clip first (supports more model variants)
        import open_clip
        model_name = "ViT-B-32"
        pretrained = "openai"

        # If user specified a HF model ID, try to map it
        if "/" in CLIP_MODEL:
            # e.g. "openai/clip-vit-base-patch32" -> use default open_clip
            if "large" in CLIP_MODEL.lower():
                model_name = "ViT-L-14"
            elif "huge" in CLIP_MODEL.lower():
                model_name = "ViT-H-14"

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=_device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        model.eval()
        _model = model
        _processor = preprocess
        _tokenizer = tokenizer
        logger.info(f"CLIP model loaded successfully ({model_name}, {pretrained}) on {_device}")
    except ImportError:
        # Fallback to transformers CLIP
        logger.info("open_clip not available, falling back to transformers CLIPModel")
        from transformers import CLIPModel, CLIPProcessor
        _model = CLIPModel.from_pretrained(CLIP_MODEL).to(_device).eval()
        _processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        _tokenizer = None  # CLIPProcessor handles tokenization
        logger.info(f"CLIP model loaded via transformers on {_device}")


def route_image(image: Image.Image, filename: str = "") -> dict:
    """Classify image type using CLIP zero-shot. Returns {type, confidence, scores}."""
    if MOCK_MODE:
        return _mock_route(filename)

    _load_model()
    import torch

    try:
        if _tokenizer is not None:
            # open_clip path
            img_tensor = _processor(image).unsqueeze(0).to(_device)
            # Use primary prompt per category
            text_labels = [PROMPTS[t][0] for t in IMAGE_TYPES]
            texts = _tokenizer(text_labels).to(_device)

            with torch.no_grad():
                img_features = _model.encode_image(img_tensor)
                txt_features = _model.encode_text(texts)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * img_features @ txt_features.T).softmax(dim=-1)[0]

            scores = {t: round(float(similarity[i]), 4) for i, t in enumerate(IMAGE_TYPES)}
        else:
            # transformers CLIPModel path
            text_labels = [PROMPTS[t][0] for t in IMAGE_TYPES]
            inputs = _processor(text=text_labels, images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = _model(**inputs)
                logits = outputs.logits_per_image[0]
                probs = logits.softmax(dim=-1)

            scores = {t: round(float(probs[i]), 4) for i, t in enumerate(IMAGE_TYPES)}

        best = max(scores, key=scores.get)
        conf = scores[best]

        # If confidence is low, try with expanded prompts (ensemble)
        if conf < 0.5 and _tokenizer is not None:
            scores = _ensemble_route(image)
            best = max(scores, key=scores.get)
            conf = scores[best]

        if conf < 0.35:
            best = "unknown"

        return {"type": best, "confidence": conf, "scores": scores}

    except Exception as e:
        logger.error(f"CLIP routing failed: {e}")
        # Fallback to filename-based routing
        result = _mock_route(filename)
        result["fallback"] = True
        return result


def _ensemble_route(image: Image.Image) -> dict:
    """Run CLIP with multiple prompts per category and average scores."""
    import torch

    all_prompts = []
    prompt_map = []  # maps prompt index -> category
    for t in IMAGE_TYPES:
        for p in PROMPTS[t]:
            all_prompts.append(p)
            prompt_map.append(t)

    img_tensor = _processor(image).unsqueeze(0).to(_device)
    texts = _tokenizer(all_prompts).to(_device)

    with torch.no_grad():
        img_features = _model.encode_image(img_tensor)
        txt_features = _model.encode_text(texts)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * img_features @ txt_features.T).softmax(dim=-1)[0]

    # Average scores per category
    cat_scores = {t: [] for t in IMAGE_TYPES}
    for i, cat in enumerate(prompt_map):
        cat_scores[cat].append(float(similarity[i]))

    scores = {t: round(sum(s) / len(s), 4) for t, s in cat_scores.items()}
    # Renormalize
    total = sum(scores.values())
    if total > 0:
        scores = {t: round(v / total, 4) for t, v in scores.items()}
    return scores


def _mock_route(filename: str) -> dict:
    fn = filename.lower()
    if any(k in fn for k in ["skin", "derm", "lesion", "mole", "nevus", "melanoma"]):
        t = "skin_lesion"
    elif any(k in fn for k in ["xray", "chest", "cxr", "lung", "pneumonia", "tb"]):
        t = "chest_xray"
    elif any(k in fn for k in ["fundus", "retina", "eye", "dr", "retinopathy"]):
        t = "fundus"
    else:
        t = random.choice(IMAGE_TYPES)
    conf = round(random.uniform(0.75, 0.97), 4)
    scores = {k: round(random.uniform(0.01, 0.15), 4) for k in IMAGE_TYPES}
    scores[t] = conf
    return {"type": t, "confidence": conf, "scores": scores}


def get_status() -> dict:
    if MOCK_MODE:
        return {"name": "CLIP Router", "status": "ready (mock)", "model": CLIP_MODEL}
    return {
        "name": "CLIP Router",
        "status": "loaded" if _model is not None else "not_loaded",
        "model": CLIP_MODEL,
        "device": _device if _model else None,
    }
