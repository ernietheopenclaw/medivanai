"""MediVan — FastAPI backend."""
import io
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from backend.config import MOCK_MODE, HOST, PORT, UPLOAD_DIR
from backend.services import router, skin_classifier, chest_classifier, eye_classifier
from backend.services import report_generator, session_manager, rag

app = FastAPI(title="MediVan", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

os.makedirs(UPLOAD_DIR, exist_ok=True)

CLASSIFIERS = {
    "skin_lesion": skin_classifier.classify,
    "chest_xray": chest_classifier.classify,
    "fundus": eye_classifier.classify,
}


async def _read_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image too large (max 10MB)")
    return Image.open(io.BytesIO(data)).convert("RGB")


@app.get("/api/health")
async def health():
    import platform
    gpu = "N/A"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except:
        pass
    return {
        "status": "healthy",
        "mock_mode": MOCK_MODE,
        "platform": platform.machine(),
        "gpu": gpu,
        "models": [
            router.get_status(),
            skin_classifier.get_status(),
            chest_classifier.get_status(),
            eye_classifier.get_status(),
        ],
    }


@app.get("/api/models")
async def models():
    return [
        router.get_status(),
        skin_classifier.get_status(),
        chest_classifier.get_status(),
        eye_classifier.get_status(),
    ]


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    image = await _read_image(file)
    route = router.route_image(image, file.filename or "")
    image_type = route["type"]

    if image_type == "unknown":
        return {"image_type": "unknown", "route": route, "result": None, "explanation": "Could not identify image type. Please upload a skin lesion, chest X-ray, or fundus photo."}

    classifier = CLASSIFIERS.get(image_type)
    result = classifier(image)
    explanation = report_generator.generate_explanation(image_type, result)
    guidelines = rag.retrieve(f"{image_type} {result['classification']}")

    return {
        "image_type": image_type,
        "route": route,
        "result": result,
        "explanation": explanation,
        "guidelines": guidelines,
    }


@app.post("/api/session/start")
async def start_session():
    return session_manager.create_session()


@app.get("/api/session/{sid}")
async def get_session(sid: str):
    s = session_manager.get_session(sid)
    if not s:
        raise HTTPException(404, "Session not found")
    return s


@app.post("/api/session/{sid}/analyze")
async def session_analyze(sid: str, file: UploadFile = File(...)):
    s = session_manager.get_session(sid)
    if not s:
        raise HTTPException(404, "Session not found")

    image = await _read_image(file)
    route = router.route_image(image, file.filename or "")
    image_type = route["type"]

    if image_type == "unknown":
        finding = {"image_type": "unknown", "classification": "unidentified", "confidence": 0, "risk_level": "low", "recommendation": "Re-upload a clearer image."}
    else:
        result = CLASSIFIERS[image_type](image)
        finding = {"image_type": image_type, **result}

    finding["route"] = route
    session_manager.add_finding(sid, finding)
    return finding


@app.post("/api/session/{sid}/report")
async def session_report(sid: str):
    s = session_manager.get_session(sid)
    if not s:
        raise HTTPException(404, "Session not found")
    if not s["findings"]:
        raise HTTPException(400, "No findings to report")
    report = report_generator.generate_report(s)
    session_manager.set_report(sid, report)
    return {"report": report}


# Serve frontend static files
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "out")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=True)
