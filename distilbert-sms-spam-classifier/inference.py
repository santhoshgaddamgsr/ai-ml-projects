from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

MODEL_PATH = "./distilbert-spam-model"

# Load tokenizer & model
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

def predict(text: str):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    label_id = torch.argmax(probs, dim=1).item()
    confidence = probs[0][label_id].item()

    label = "spam" if label_id == 1 else "ham"

    return {
        "label": label,
        "confidence": round(confidence, 4)
    }

# quick test
if __name__ == "__main__":
    print(predict("Congratulations! You won a free prize"))
