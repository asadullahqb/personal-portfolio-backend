import os
import requests
from typing import Dict, Any

# --- Configuration ---
AZURE_MAPS_KEY = os.environ.get("AZURE_MAPS_SUBSCRIPTION_KEY")
AZURE_MAPS_URL = "https://atlas.microsoft.com/geolocation/ip/json"
API_VERSION = "1.0"
BASE_WELCOME_MESSAGE = "Welcome."
DEFAULT_LANGUAGE_CODE = "en"

# Hugging Face Inference API Configuration
HF_API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen1.5-7B-Instruct"
HF_API_KEY = os.environ.get("HF_API_KEY")  # Must be set in environment

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

def instruct_get_localized_welcome(country_code: str) -> Dict[str, str]:
    """
    Uses Qwen1.5-7B-Instruct (via Hugging Face API) to directly return both primary language code and localized welcome.
    """
    if country_code == "DEFAULT" or not HF_API_KEY:
        return {
            "language": DEFAULT_LANGUAGE_CODE,
            "message": BASE_WELCOME_MESSAGE
        }
    
    prompt = (
        f"You are an international greeter bot. "
        f"Given a 2-letter ISO country code, output a JSON object with two keys: "
        f'"language" (the two-letter ISO 639-1 primary language code of that country) and '
        f'"message" (a translation of \'Welcome.\' appropriate for that country and language, with only the welcome word or phrase, no extra text). '
        f"Only output the JSON. Country code: {country_code}"
    )

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.2,
            "max_new_tokens": 96,
            "return_full_text": False
        }
    }
    try:
        resp = requests.post(HF_API_URL, json=payload, headers=headers, timeout=18)
        resp.raise_for_status()
        hf_payload = resp.json()
        # "generated_text" may be returned as a dict or inside a list
        generated = None
        if isinstance(hf_payload, list) and hf_payload and "generated_text" in hf_payload[0]:
            generated = hf_payload[0]["generated_text"]
        elif isinstance(hf_payload, dict) and "generated_text" in hf_payload:
            generated = hf_payload["generated_text"]
        import json as _json
        response_obj = None
        if generated:
            import re as _re
            match = _re.search(r"\{[\s\S]*?\}", generated)
            if match:
                generated = match.group(0)
            try:
                response_obj = _json.loads(generated)
            except Exception as e:
                print(f"Could not parse instruct JSON output: '{generated}' :: {e}")
        if (
            response_obj
            and isinstance(response_obj, dict)
            and "message" in response_obj
            and "language" in response_obj
        ):
            return {
                "language": response_obj["language"].strip().lower(),
                "message": response_obj["message"].strip()
            }
        else:
            print(f"Qwen instruct response did not contain valid keys: {hf_payload}")
    except Exception as e:
        print(f"Hugging Face instruct API error: {e}")
    return {
        "language": DEFAULT_LANGUAGE_CODE,
        "message": BASE_WELCOME_MESSAGE
    }

def get_welcome_message(ip: str) -> Dict[str, Any]:
    """
    1. Gets country code from IP via Azure Maps.
    2. Uses a single Qwen1.5-7B-Instruct call to get both language code and localized welcome message.
    """
    country_code = get_country_code_from_ip(ip)
    result = instruct_get_localized_welcome(country_code)
    return {
        "message": result["message"],
        "language": result["language"],
        "country_code": country_code,
        "ip_used": ip,
        "source": "Azure Maps + Qwen1.5-7B-Instruct"
    }