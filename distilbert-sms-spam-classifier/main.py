import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from google.cloud import storage

# -------------------------
# FastAPI app (DESCRIPTION)
# -------------------------
app = FastAPI(
    title="DistilBERT Spam Classifier",
    description="""
API to classify SMS / text messages as **Spam** or **Ham** using a trained DistilBERT model.

### Sample inputs

**Spam examples**
- Congratulations! You have won a free lottery ticket. Claim now.
- Win ₹10,000 today. Limited time offer.
- Free recharge offer! Act now to get cashback.

**Ham examples**
- Hey, are we meeting tomorrow at 10 am?
- Please review the attached report before the meeting.
- Can you call me when you are free?

### How to test
1. Open **POST /predict**
2. Click **Try it out**
3. Edit or replace the pre-filled example  
4. You can also enter **your own custom text or message**
5. Click **Execute** to see the prediction
""",
    version="1.0"
)

# -------------------------
# GCS configuration
# -------------------------
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_MODEL_PREFIX = "distilbert-spam/production/distilbert-spam-model"
LOCAL_MODEL_DIR = os.path.join(os.getenv("TEMP", "/tmp"), "distilbert-spam-model")


tokenizer = None
model = None


def download_model_from_gcs():
    """Download model files from GCS to /tmp (Cloud Run writable)."""
    if os.path.exists(os.path.join(LOCAL_MODEL_DIR, "config.json")):
        return

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    blobs = bucket.list_blobs(prefix=GCS_MODEL_PREFIX)

    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        relative_path = blob.name.replace(GCS_MODEL_PREFIX + "/", "")
        local_path = os.path.join(LOCAL_MODEL_DIR, relative_path)

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)


# -------------------------
# LAZY MODEL LOADING (Cloud Run safe)
# -------------------------
def load_model():
    """
    Loads the tokenizer and model only once,
    on the first request.
    Prevents Cloud Run startup crash.
    """
    global tokenizer, model

    if model is not None and tokenizer is not None:
        return

    download_model_from_gcs()

    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_DIR,
        use_fast=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        LOCAL_MODEL_DIR
    )

    # Explicit CPU usage (Cloud Run has no GPU)
    model.to("cpu")
    model.eval()


# -------------------------
# Request schema
# -------------------------
class TextRequest(BaseModel):
    text: str = Field(
        ...,
        example="Congratulations! You have won a free lottery ticket. Claim now."
    )


# -------------------------
# POST endpoint (UP)
# -------------------------
@app.post("/predict", summary="Predict spam or ham")
def predict_spam(req: TextRequest):
    load_model()

    inputs = tokenizer(
        req.text,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    label_id = torch.argmax(probs, dim=1).item()
    confidence = probs[0][label_id].item()

    return {
        "input_text": req.text,
        "label": "spam" if label_id == 1 else "ham",
        "confidence": round(confidence, 4)
    }


# -------------------------
# GET endpoint (DOWN)
# -------------------------
@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}
