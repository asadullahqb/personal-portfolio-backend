import os
import requests
from typing import Optional

# 1. Load the Secret into environment variables
# This pulls the 'AZURE_MAPS_SUBSCRIPTION_KEY' from the Secrets pane.
AZURE_MAPS_KEY = os.environ.get("AZURE_MAPS_SUBSCRIPTION_KEY")
AZURE_MAPS_URL = "https://atlas.microsoft.com/geolocation/ip/json"
API_VERSION = "1.0"

# --- Configuration ---
# Now it safely reads from os.environ
AZURE_MAPS_KEY = os.environ.get("AZURE_MAPS_SUBSCRIPTION_KEY")
AZURE_MAPS_URL = "https://atlas.microsoft.com/geolocation/ip/json"
API_VERSION = "1.0"

# --- Translation Data ---
TRANSLATIONS = {
    "US": "Welcome", "GB": "Welcome", "FR": "Bienvenue",
    "ES": "Bienvenido", "DE": "Willkommen", "CN": "欢迎",
    "JP": "ようこそ", "IN": "स्वागत है", "MY": "Selamat Datang",
    "default": "Welcome"
}

# --- Azure Maps Integration Function ---

def get_country_from_ip(ip: str) -> Optional[str]:
    """Uses Azure Maps Geolocation API to get the 2-letter country code."""
    if not AZURE_MAPS_KEY:
        print("ERROR: Azure Maps key is missing. Using default.")
        return TRANSLATIONS["default"]

    params = {
        'api-version': API_VERSION,
        'subscription-key': AZURE_MAPS_KEY,
        'ip': ip 
    }

    try:
        response = requests.get(AZURE_MAPS_URL, params=params, timeout=5)
        response.raise_for_status() 
        
        data = response.json()
        country_code = data.get('countryRegion', {}).get('isoCode')
        
        if country_code:
            return country_code.upper()
            
    except requests.exceptions.RequestException as e:
        print(f"Azure Maps API Error: {e}. Using default.")
        
    except KeyError:
        print("Azure Maps response structure error. Using default.")

    return TRANSLATIONS["default"]


def translate_welcome(ip: str) -> str:
    """Translates the welcome message based on the IP-derived country code."""
    country = get_country_from_ip(ip)
    return TRANSLATIONS.get(country, TRANSLATIONS["default"])