"""CLIP/SigLIP zero-shot image router."""
import random
from PIL import Image
from backend.config import MOCK_MODE, CLIP_MODEL

IMAGE_TYPES = ["skin_lesion", "chest_xray", "fundus"]
PROMPTS = {
    "skin_lesion": "a dermoscopic image of a skin lesion",
    "chest_xray": "a chest x-ray radiograph",
    "fundus": "a fundus photograph of the retina",
}

_model = None
_processor = None
_tokenizer = None


def _load_model():
    global _model, _processor, _tokenizer
    if _model is not None:
        return
    import open_clip
    import torch
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    _model = model
    _processor = preprocess
    _tokenizer = tokenizer


def route_image(image: Image.Image, filename: str = "") -> dict:
    """Classify image type. Returns {type, confidence, scores}."""
    if MOCK_MODE:
        return _mock_route(filename)

    _load_model()
    import torch

    img_tensor = _processor(image).unsqueeze(0)
    texts = _tokenizer(list(PROMPTS.values()))

    with torch.no_grad():
        img_features = _model.encode_image(img_tensor)
        txt_features = _model.encode_text(texts)
        img_features /= img_features.norm(dim=-1, keepdim=True)
        txt_features /= txt_features.norm(dim=-1, keepdim=True)
        similarity = (img_features @ txt_features.T).softmax(dim=-1)[0]

    scores = {t: float(similarity[i]) for i, t in enumerate(IMAGE_TYPES)}
    best = max(scores, key=scores.get)
    conf = scores[best]
    if conf < 0.4:
        best = "unknown"
    return {"type": best, "confidence": round(conf, 4), "scores": scores}


def _mock_route(filename: str) -> dict:
    fn = filename.lower()
    if any(k in fn for k in ["skin", "derm", "lesion", "mole"]):
        t = "skin_lesion"
    elif any(k in fn for k in ["xray", "chest", "cxr", "lung"]):
        t = "chest_xray"
    elif any(k in fn for k in ["fundus", "retina", "eye", "dr"]):
        t = "fundus"
    else:
        t = random.choice(IMAGE_TYPES)
    conf = round(random.uniform(0.75, 0.97), 4)
    scores = {k: round(random.uniform(0.01, 0.15), 4) for k in IMAGE_TYPES}
    scores[t] = conf
    return {"type": t, "confidence": conf, "scores": scores}


def get_status() -> dict:
    if MOCK_MODE:
        return {"name": "CLIP Router", "status": "mock", "model": CLIP_MODEL}
    return {
        "name": "CLIP Router",
        "status": "loaded" if _model is not None else "not_loaded",
        "model": CLIP_MODEL,
    }
