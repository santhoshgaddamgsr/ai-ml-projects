# DistilBERT SMS Spam Classifier

This project implements a **DistilBERT-based NLP model** for classifying SMS/text messages as **Spam** or **Ham**.

The solution follows a **production-style inference design**, where trained model artifacts are managed externally and loaded dynamically during prediction.

---

## Overview
- Binary text classification (Spam vs Ham)
- Fine-tuned DistilBERT transformer
- Designed for real-world inference usage
- Model files intentionally kept outside source control

---

## Architecture
```
Client Request
|
v
FastAPI Inference API
|
v
DistilBERT Model (CPU)
|
v
Prediction + Confidence Score
```

Model artifacts are stored externally and loaded at runtime.

---

## Features
- Transformer-based text classification using DistilBERT
- Spam vs Ham prediction
- Confidence score returned with each prediction
- CPU-only inference support
- Lazy model loading for stability

---

## Example Predictions
- "Win ₹10,000 today. Limited time offer!" → Spam  
- "Are we meeting tomorrow morning?" → Ham  

---

## Project Structure
- `main.py` – FastAPI inference service
- `inference.py` – prediction logic
- `Dockerfile` – container configuration
- `requirements.txt` – dependencies

---

## Tech Stack
- Python
- HuggingFace Transformers
- PyTorch (CPU)
- FastAPI

---

## Note
Trained model files are intentionally excluded from this repository and are loaded separately during inference.
