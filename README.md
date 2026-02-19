# 🏥 MediVan — Mobile Diagnostic Hub

**Offline multi-modal medical diagnostic screening for underserved communities.**

MediVan AI turns a Dell Pro Max GB10 into a portable AI-powered screening station. A clinician in a mobile health van connects their phone via Tailscale and captures medical images. On-device AI models identify the image type, classify findings, and generate a holistic patient report — all without internet access.

> 🏆 Built for the Dell Pro Max GB10 Hackathon at NYU Center for Data Science

---

## 🎯 Why MediVan AI?

- **60M+ Americans** live in medically underserved areas
- Mobile health vans bridge the gap — but lack diagnostic AI
- MediVan AI brings specialist-level screening to the point of care
- **Zero cloud dependency** — all inference runs on the GB10
- **HIPAA-aligned** — patient images never leave the device

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│  📱 Phone Browser (PWA via Tailscale)               │
│  Camera capture / Image upload                      │
└──────────────────┬──────────────────────────────────┘
                   │ HTTPS (Tailscale)
┌──────────────────▼──────────────────────────────────┐
│  🖥️ Dell Pro Max GB10                               │
│                                                     │
│  FastAPI Server (:8000)                             │
│  ┌─────────────────────────────────────────────┐    │
│  │ 1. CLIP/SigLIP Image Router (zero-shot)     │    │
│  │    "skin lesion" vs "chest xray" vs "fundus" │    │
│  └──────────┬──────────┬──────────┬────────────┘    │
│             ▼          ▼          ▼                  │
│  ┌─────────────┐ ┌──────────┐ ┌──────────┐         │
│  │ 🔬 Skin ViT │ │ 🫁 CXR   │ │ 👁️ DR    │         │
│  │ HAM10000    │ │ ViT      │ │ ViT      │         │
│  │ 7 classes   │ │ 7 classes│ │ 5 grades │         │
│  └──────┬──────┘ └────┬─────┘ └────┬─────┘         │
│         └──────────────┼───────────-┘               │
│                        ▼                            │
│  ┌─────────────────────────────────────────────┐    │
│  │ 📚 RAG: Clinical Guidelines (FAISS local)   │    │
│  └──────────────────┬──────────────────────────┘    │
│                     ▼                               │
│  ┌─────────────────────────────────────────────┐    │
│  │ 🧠 LLM (NIM): Holistic Patient Report       │    │
│  │    Llama 3.1 8B via NVIDIA NIM              │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

---

## 🔧 NVIDIA Software Stack

| Component | Technology |
|-----------|-----------|
| Image Router | OpenCLIP (ViT-B/32) — zero-shot classification |
| Skin Classifier | ViT-Large/32-384 fine-tuned on HAM10000 |
| Chest X-ray | ViT fine-tuned on chest X-ray datasets |
| Eye/DR | ViT fine-tuned on diabetic retinopathy |
| Clinical LLM | Llama 3.1 8B via **NVIDIA NIM** |
| RAG Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS (CPU) |
| GPU Runtime | NVIDIA Container Toolkit / PyTorch |

---

## 💾 Memory Budget (GB10 — 128GB unified RAM)

| Model | ~VRAM | Notes |
|-------|-------|-------|
| CLIP ViT-B/32 | ~0.6 GB | Image routing |
| Skin ViT-L/32 | ~1.2 GB | HAM10000 classifier |
| Chest ViT | ~0.3 GB | X-ray classifier |
| DR ViT | ~0.3 GB | Retinopathy grading |
| Llama 3.1 8B (NIM) | ~16 GB | Report generation |
| FAISS + Embeddings | ~0.5 GB | RAG retrieval |
| **Total** | **~19 GB** | Well within 128GB budget |

---

## 🚀 Quick Start

### Prerequisites
- Dell Pro Max GB10 (ARM64) or any Linux/macOS/Windows with Python 3.10+
- NVIDIA GPU (optional — works in mock mode without GPU)
- Node.js 18+ (for frontend build)

### 1. Clone & Install

```bash
git clone https://github.com/ernietheopenclaw/medivanai.git
cd medivanai

# Backend
pip install -r backend/requirements.txt

# Frontend
cd frontend && npm install && npm run build && cd ..
```

### 2. Run (Mock Mode — no GPU needed)

```bash
MOCK_MODE=true python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 3. Run (Full — with models)

```bash
MOCK_MODE=false python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 4. Docker (GB10 deployment)

```bash
docker compose up --build
```

---

## 📱 Phone Access via Tailscale

1. Install Tailscale on the GB10: `bash scripts/setup-tailscale.sh`
2. Install Tailscale on your phone (iOS/Android)
3. Join the same Tailnet (same account)
4. Open `http://<gb10-tailscale-ip>:8000` in Chrome
5. Add to Home Screen for PWA experience

Tailscale provides:
- **Secure P2P connection** — no public internet needed
- **Valid HTTPS certs** — enables camera access on phone
- **Zero config networking** — works anywhere

---

## 📋 Demo Flow

1. **Open MediVan AI** on phone browser
2. **Start Session** — creates a new patient screening
3. **Capture/Upload** a skin lesion photo → AI classifies (e.g., "melanocytic nevi, 87% confidence, Low Risk")
4. **Capture/Upload** a chest X-ray → AI classifies (e.g., "pneumonia, 82% confidence, High Risk")
5. **Capture/Upload** a fundus photo → AI grades DR (e.g., "Moderate NPDR, 78% confidence")
6. **Generate Report** — LLM synthesizes all findings into a holistic patient report with cross-modality insights and priority referrals

---

## 🔒 Privacy & HIPAA

- **All inference runs on-device** — no cloud API calls
- **Patient images stored in volatile memory** — cleared on restart
- **No telemetry or logging of PHI**
- **Tailscale connection** — encrypted P2P, no data traverses public internet
- **Session data** — in-memory only, not persisted to disk

---

## 📁 Project Structure

```
medivanai/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── config.py             # Configuration
│   ├── services/
│   │   ├── router.py         # CLIP image router
│   │   ├── skin_classifier.py
│   │   ├── chest_classifier.py
│   │   ├── eye_classifier.py
│   │   ├── report_generator.py  # NIM LLM integration
│   │   ├── rag.py            # FAISS + guidelines
│   │   └── session_manager.py
│   ├── knowledge/            # Clinical guidelines (MD)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                 # Next.js PWA
│   ├── src/app/              # Pages
│   └── src/components/       # UI components
├── scripts/
│   ├── setup-tailscale.sh
│   └── start.sh
├── docker-compose.yml
└── README.md
```

---

## ⚠️ Disclaimer

MediVan AI is an **AI-assisted screening tool** for research and demonstration purposes. It does **NOT** constitute medical diagnosis. All findings must be confirmed by a qualified healthcare provider. Clinical decisions should be based on comprehensive evaluation, not AI screening alone.

---

## 📄 License

MIT — see [LICENSE](LICENSE)

