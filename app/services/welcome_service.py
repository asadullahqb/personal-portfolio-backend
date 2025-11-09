import os
import requests
import torch
import re
from typing import Dict, Any
# CRITICAL: Using the correct function names from your model_loader.py
from app.utils.model_loader import load_translator_model, load_global_language_inferencer

# --- Configuration ---
AZURE_MAPS_KEY = os.environ.get("AZURE_MAPS_SUBSCRIPTION_KEY")
AZURE_MAPS_URL = "https://atlas.microsoft.com/geolocation/ip/json"
API_VERSION = "1.0"
BASE_WELCOME_MESSAGE = "Welcome."
DEFAULT_LANGUAGE_CODE = "en" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_country_code_from_ip(ip: str) -> str:
    """
    Uses Azure Maps Geolocation API to convert IP to a 2-letter Country Code (ISO-3166).
    """
    if not AZURE_MAPS_KEY:
        print("WARNING: AZURE_MAPS_SUBSCRIPTION_KEY is not set. Using default country.")
        return "DEFAULT" 

    params = {
        'api-version': API_VERSION,
        'subscription-key': AZURE_MAPS_KEY,
        'ip': ip 
    }

    try:
        response = requests.get(AZURE_MAPS_URL, params=params, timeout=3)
        response.raise_for_status()
        data = response.json()
        country_code = data.get('countryRegion', {}).get('isoCode')
        
        if country_code:
            return country_code.upper()
            
    except requests.exceptions.RequestException as e:
        print(f"Azure Maps API Error: {e}. Falling back to default country.")

    return "DEFAULT"

def infer_language_from_country(country_code: str) -> str:
    """
    Uses the Inference AI model (T5-small) to determine the correct ISO language code.
    """
    # CRITICAL FIX 1: If Azure Maps failed, bypass the unreliable AI inference
    if country_code == "DEFAULT":
        print("Bypassing AI inference for 'DEFAULT' country. Using English fallback code.")
        return DEFAULT_LANGUAGE_CODE

    tokenizer, model = load_global_language_inferencer()
    
    if model is None:
        print("Language Inference Model failed to load. Defaulting to English.")
        return DEFAULT_LANGUAGE_CODE
    
    prompt = f"What is the two-letter ISO 639-1 language code for the primary language spoken in the country with the code {country_code}? Output only the two-letter code."

    try:
        # Check model device
        if model.device != DEVICE:
            model = model.to(DEVICE)

        inputs = tokenizer([prompt], return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=10, 
            do_sample=False,
            num_beams=2
        )
        
        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
        
        match = re.search(r'([a-z]{2})', raw_output)
        
        if match:
            lang_code = match.group(1)
            print(f"AI inferred language code (cleaned): {lang_code}")
            return lang_code
        else:
            print(f"AI inferred an unreliable code '{raw_output}' for {country_code}. Defaulting.")
            return DEFAULT_LANGUAGE_CODE

    except Exception as e:
        print(f"Language Inference AI failed: {e}. Defaulting to English.")
        return DEFAULT_LANGUAGE_CODE

def get_welcome_message(ip: str) -> Dict[str, Any]:
    """
    1. Gets country code from IP via Azure Maps.
    2. Uses Inference AI to get the language code.
    3. Uses Translation AI (Opus-MT) to translate the message.
    """
    tokenizer_t, model_t = load_translator_model()
    
    if model_t is None:
        return {"message": "Welcome (Translation Model Offline)", "language": DEFAULT_LANGUAGE_CODE, "ip_used": ip}
        
    country_code = get_country_code_from_ip(ip)
    # FIX: target_lang will now be 'en' if country_code is 'DEFAULT'
    target_lang = infer_language_from_country(country_code)
    
    # 3. Construct Opus-MT Prompt: >>lang<< text (Required format for Opus-MT)
    target_tag = f">>{target_lang}<<"
    prompt = f"{target_tag} {BASE_WELCOME_MESSAGE}"

    print(f"Final Prompt: {prompt}")
    
    inputs = tokenizer_t([prompt], return_tensors="pt", max_length=512, truncation=True)
    # CRITICAL FIX: Ensure inputs are on the device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()} 

    # 5. Generate Translation
    try:
        outputs = model_t.generate(
            **inputs, 
            max_new_tokens=50, 
            do_sample=False,
            num_beams=4,
            early_stopping=True
        )
        
        translation = tokenizer_t.decode(outputs[0], skip_special_tokens=True).strip()

        return {
            "message": translation, 
            "language": target_lang,
            "country_code": country_code,
            "ip_used": ip,
            "source": "Azure Maps + Two-Stage AI"
        }

    except Exception as e:
        print(f"Translation AI failed: {e}. Returning English fallback.")
        return {
            "message": BASE_WELCOME_MESSAGE, 
            "language": DEFAULT_LANGUAGE_CODE,
            "country_code": country_code,
            "ip_used": ip,
            "source": "Fallback (AI Error)"
        }