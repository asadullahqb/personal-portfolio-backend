from app.utils.model_loader import load_model_b
import torch

def predict(text: str):
    model, tokenizer = load_model_b()
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    pred_label = torch.argmax(outputs.logits, dim=1).item()
    return {"label": str(pred_label)}
