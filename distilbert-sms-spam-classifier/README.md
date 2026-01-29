# DistilBERT SMS Spam Classifier

This project is a **DistilBERT-based NLP model** for classifying SMS/text messages as **Spam** or **Ham**.

The model is served using **FastAPI** and designed in a **production-style setup**, where trained model artifacts are stored externally and loaded at inference time.

---

## Features
- DistilBERT transformer for text classification
- FastAPI-based inference API
- CPU-only inference (no GPU required)
- Lazy model loading for stable startup
- Confidence score returned with prediction

---

## API Endpoints

### POST `/predict`
Classifies a message as spam or ham.

**Request**
```json
{
  "text": "Congratulations! You have won a free lottery ticket."
}
```
Response
```
{
  "label": "spam",
  "confidence": 0.98
}
```
##  GET /health
Health check endpoint.

## Tech Stack
- Python
- HuggingFace Transformers
- PyTorch (CPU)
- FastAPI

## Note
Trained model files are intentionally excluded from this repository and are loaded separately during inference.
