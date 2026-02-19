"""MediVan AI configuration."""
import os

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() in ("true", "1", "yes")
NIM_ENDPOINT = os.getenv("NIM_ENDPOINT", "http://localhost:8080/v1")
NIM_MODEL = os.getenv("NIM_MODEL", "meta/llama-3.1-8b-instruct")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/medivanai_uploads")
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB

# Model paths (HuggingFace model IDs)
CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
SKIN_MODEL = os.getenv("SKIN_MODEL", "nickjourjine/vit-large-patch32-384-finetuned-HAM10000")
CHEST_MODEL = os.getenv("CHEST_MODEL", "codewithdark/vit-chest-xray")
EYE_MODEL = os.getenv("EYE_MODEL", "rafalosa/diabetic-retinopathy-224-procnorm-vit")

# RAG
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")

