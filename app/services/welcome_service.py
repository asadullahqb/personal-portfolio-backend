from app.utils.model_loader import load_translator_model

def get_welcome_message(ip: str) -> dict:
    tokenizer, model = load_translator_model()
    
    # Prompt AI to translate 'Welcome' based on IP
    prompt = f"Given the user IP {ip}, translate the word 'Welcome' into the local language."
    inputs = tokenizer([prompt], return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=20)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"message": translation}
