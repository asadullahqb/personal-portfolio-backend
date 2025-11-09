from app.utils.model_loader import load_translator_model
import geoip2.database

# Load GeoLite2 database once
reader = geoip2.database.Reader("GeoLite2-City.mmdb")

def get_welcome_message(ip: str) -> dict:
    # 1. Get country from IP
    try:
        response = reader.city(ip)
        country = response.country.name
    except Exception:
        country = "Unknown"

    # 2. Load AI translation model
    tokenizer, model = load_translator_model()

    # 3. Prompt AI: Map country -> language and translate 'Welcome'
    prompt = (
        f"Given the country '{country}', determine the main language spoken there. "
        f"Then translate the word 'Welcome' into that language. "
        "Return only the translated word."
    )
    inputs = tokenizer([prompt], return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=20)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"message": translation}
