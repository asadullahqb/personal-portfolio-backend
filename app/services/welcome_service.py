import os
import requests
import torch
from typing import Dict, Any
from app.utils.model_loader import get_global_translator

# --- Configuration & Mapping ---
# AZURE MAPS KEY: Read from the Uvicorn environment
AZURE_MAPS_KEY = os.environ.get("AZURE_MAPS_SUBSCRIPTION_KEY")
AZURE_MAPS_URL = "https://atlas.microsoft.com/geolocation/ip/json"
API_VERSION = "1.0"

BASE_WELCOME_MESSAGE = "Welcome."
DEFAULT_LANGUAGE_CODE = "en" # Still need a fallback code for error reporting

# --- Azure Maps Logic (IP to Country Code) ---
# We keep this dedicated function because Azure Maps is the authoritative source for IP geolocation.

def get_country_code_from_ip(ip: str) -> str:
    """
    Uses Azure Maps Geolocation API to convert IP to a 2-letter Country Code (ISO-3166).
    Returns 'default' string if the API call fails or the key is missing.
    """
    if not AZURE_MAPS_KEY:
        print("WARNING: AZURE_MAPS_SUBSCRIPTION_KEY is not set. Cannot perform API lookup.")
        return "DEFAULT" # Return a fallback string that won't confuse the AI

    params = {
        'api-version': API_VERSION,
        'subscription-key': AZURE_MAPS_KEY,
        'ip': ip 
    }

    try:
        # Use requests for server-side HTTP call to Azure Maps
        response = requests.get(AZURE_MAPS_URL, params=params, timeout=3)
        response.raise_for_status()
        data = response.json()
        country_code = data.get('countryRegion', {}).get('isoCode')
        
        if country_code:
            return country_code.upper()
            
    except requests.exceptions.RequestException as e:
        print(f"Azure Maps API Error: {e}. Falling back to default country.")

    # Fallback returns a non-country string
    return "DEFAULT"

# --- Combined Logic (Azure Maps + AI) ---

def get_welcome_message(ip: str) -> Dict[str, Any]:
    """
    1. Gets country code from IP via Azure Maps.
    2. Prompts the Hugging Face model to infer the language and translate the message.
    """
    tokenizer, model = get_global_translator()
    
    # Fallback if model failed to load
    if model is None:
        return {"message": "Welcome (Model Offline)", "language": DEFAULT_LANGUAGE_CODE, "ip_used": ip}

    # 1. Get Country Code (e.g., "FR")
    country_code = get_country_code_from_ip(ip)
    
    # 2. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model.device != device:
        model = model.to(device)

    # 3. Construct AI Prompt (AI performs language inference and translation)
    # The prompt MUST be highly specific to ensure the T5 model gives the right output.
    if country_code == "DEFAULT":
        # If API failed, ask the AI to translate to English as a default
        prompt = f"translate English to English: {BASE_WELCOME_MESSAGE}"
    else:
        # Ask the model to infer the correct language from the country code and translate
        # We specify 'English' as the source for the T5 model.
        prompt = f"translate English to the primary language of the country with ISO code {country_code}: {BASE_WELCOME_MESSAGE}"

    print(f"AI Prompt for IP {ip} (Country: {country_code}): {prompt}")
    
    # 4. Tokenize Input
    inputs = tokenizer([prompt], return_tensors="pt", max_length=512, truncation=True).to(device)

    # 5. Generate Translation
    try:
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50, 
            do_sample=False,
            num_beams=4,
            early_stopping=True
        )
        
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # NOTE: The T5 model won't return the language code, so we infer 'language' as unknown
        return {
            "message": translation, 
            "language": f"Inferred by AI from {country_code}",
            "ip_used": ip,
            "source": "Azure Maps + Hugging Face AI"
        }

    except Exception as e:
        print(f"Hugging Face translation failed: {e}. Returning English fallback.")
        return {
            "message": BASE_WELCOME_MESSAGE, 
            "language": DEFAULT_LANGUAGE_CODE,
            "ip_used": ip,
            "source": "Fallback (AI Error)"
        }