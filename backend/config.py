"""MediVan AI configuration."""
import os
import logging

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[MediVan AI] %(levelname)s %(name)s: %(message)s",
)

# ── Mode ──────────────────────────────────────────────────
# Set MOCK_MODE=false to use real models (requires PyTorch + HuggingFace models downloaded)
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() in ("true", "1", "yes")

# ── Server ────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/medivanai_uploads")
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB

# ── LLM (OpenAI-compatible API: NIM, Ollama, vLLM, etc.) ─
NIM_ENDPOINT = os.getenv("NIM_ENDPOINT", "http://localhost:8080/v1")
NIM_MODEL = os.getenv("NIM_MODEL", "meta/llama-3.1-8b-instruct")

# ── CV Models (HuggingFace model IDs) ────────────────────
# Image Router — CLIP zero-shot classifier
CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")

# Skin Lesion Classifier — ViT fine-tuned on HAM10000
SKIN_MODEL = os.getenv("SKIN_MODEL", "nickjourjine/vit-large-patch32-384-finetuned-HAM10000")

# Chest X-ray Classifier — ViT fine-tuned on ChestX-ray14
CHEST_MODEL = os.getenv("CHEST_MODEL", "codewithdark/vit-chest-xray")

# Diabetic Retinopathy Classifier — ViT fine-tuned on APTOS/EyePACS
EYE_MODEL = os.getenv("EYE_MODEL", "rafalosa/diabetic-retinopathy-224-procnorm-vit")

# ── RAG ───────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")

# ── GPU ───────────────────────────────────────────────────
# Auto-detected at model load time. Set CUDA_VISIBLE_DEVICES to control GPU selection.
# On GB10: unified 128GB memory, no need to manage GPU/CPU split.
