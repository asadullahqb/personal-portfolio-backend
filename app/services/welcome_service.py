from app.utils.model_loader import load_translator_model
import torch

def get_welcome_message(ip: str) -> dict:
    tokenizer, model = load_translator_model()

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Prompt AI to translate 'Welcome' based on IP
    prompt = f"Given the user IP {ip}, translate the word 'Welcome' into the local language."
    inputs = tokenizer([prompt], return_tensors="pt").to(device)  # Move inputs to same device

    # Generate translation
    outputs = model.generate(**inputs, max_new_tokens=20)

    # Decode output
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"message": translation}
