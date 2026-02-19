# Example Images for MediVan AI

Test images for each screening modality. Some are real (from Wikimedia Commons, CC-licensed), others are placeholders — replace with real dataset images for best results.

## 🔬 Skin Lesions (`skin/`)
- `melanoma_example.jpg` — Real melanoma image (Wikimedia Commons, CC)
- `benign_nevus_example.jpg` — Placeholder (replace with HAM10000 image)

**Download real images:** [HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) or [ISIC Archive](https://www.isic-archive.com/)

## 🫁 Chest X-Ray (`chest_xray/`)
- `normal_cxr.png` — Real normal chest X-ray (Wikimedia Commons, CC)
- `pneumonia_cxr.jpg` — Placeholder (replace with ChestX-ray14 image)

**Download real images:** [ChestX-ray14 on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) or [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC)

## 👁️ Fundus / Retinopathy (`fundus/`)
- `normal_fundus.jpg` — Placeholder (replace with EyePACS/APTOS image)
- `diabetic_retinopathy.jpg` — Placeholder (replace with APTOS image)

**Download real images:** [APTOS 2019 on Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection/data) or [Messidor-2](https://www.adcis.net/en/third-party/messidor2/)

## Usage
```bash
# Test the image router
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@examples/skin/melanoma_example.jpg"

# Test chest X-ray classification
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@examples/chest_xray/normal_cxr.png"
```

## For the Hackathon
Download real images from the datasets above onto a flash drive **before** the hackathon. The HAM10000 dataset (~2GB) and a subset of ChestX-ray14 will give you hundreds of test images across all categories.
