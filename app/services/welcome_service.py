import os
import requests
from typing import Dict, Any, Optional
from collections import defaultdict
import threading
import ipaddress
import json as _json
import re as _re

# --- Environment & Configuration ---

# Check if running in development mode for conditional .env loading
current_env = os.environ.get('VERCEL_ENV', 'development')

# Conditional dotenv loading MUST happen before any code tries to access the key
if current_env == 'development':
    # Import is placed here to avoid ImportError if python-dotenv is not installed in prod
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("Environment: Local Development. Loaded variables from .env file.")
    except ImportError:
        print("WARNING: python-dotenv not installed. Using system variables only.")
else:
    print(f"Environment: {current_env.capitalize()}. Using Vercel environment variables.")

API_URL = "https://router.huggingface.co/v1/chat/completions"

# Global placeholder for the key, retrieved safely inside query()
_HF_API_KEY_CACHE: Optional[str] = None 

# Thread-safe in-memory cache: {ip_prefix (first 3 octets): {country_code: {message, language}}}
_ip_cache = {}
_cache_lock = threading.Lock()


def _get_hf_headers() -> Optional[Dict[str, str]]:
    """Retrieves the API key safely and returns the header dict."""
    global _HF_API_KEY_CACHE
    
    if not _HF_API_KEY_CACHE:
        # FIX 1: Use .get() to avoid crashing on startup (KeyError)
        key = os.environ.get("HF_API_KEY")
        if not key:
            print("ERROR: HF_API_KEY is missing from environment variables.")
            return None
        _HF_API_KEY_CACHE = key

    return {
        "Authorization": f"Bearer {_HF_API_KEY_CACHE}",
    }


def _ip_prefix(ip: str, prefix_octets: int = 3) -> str:
    """Return the first N octets of the IPv4 address as a prefix string."""
    try:
        parts = ip.split('.')
        if len(parts) != 4:
            # We don't handle IPv6 here, return empty string which will bypass cache
            return ""
        return '.'.join(parts[:prefix_octets])
    except Exception:
        return ""


def _find_in_cache(ip: str):
    """Try to find a cached welcome for similar (same /24) IP."""
    prefix = _ip_prefix(ip)
    with _cache_lock:
        return _ip_cache.get(prefix)


def _add_to_cache(ip: str, country_code: str, language: str, message: str):
    prefix = _ip_prefix(ip)
    # Store whole result under this prefix
    with _cache_lock:
        _ip_cache[prefix] = {
            "country_code": country_code,
            "language": language,
            "message": message,
        }

def query(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Queries the Hugging Face API."""
    headers = _get_hf_headers()
    if not headers:
        return None # Return None if key is missing
        
    response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
    response.raise_for_status()
    return response.json()


def instruct_get_localized_welcome_from_ip(ip: str) -> Dict[str, str]:
    """
    Uses an LLM (via Hugging Face API) to determine: Country, Language, and Message.
    """
    # Try from cache first
    cached = _find_in_cache(ip)
    if cached:
        # Compose our response as required, add 'source'
        return {
            "language": cached["language"].strip().lower(),
            "message": cached["message"].strip(),
            "country_code": cached["country_code"].strip().upper(),
            "source": "cache"
        }

    prompt = (
        f"You are an international greeter bot. "
        f"Given an IPv4 address, infer the most probable country. "
        f"Output a JSON object with exactly three keys: "
        f'"country_code" (the two-letter ISO 3166-1 code of the country most likely associated with the IP), '
        f'"language" (the two-letter ISO 639-1 primary language code for that country), and '
        f'"message" (a translation of \'Welcome.\' appropriate for that country and language, with only the word or phrase, and a period at the end, no extra text). '
        f"Only output the JSON. IP address: {ip}"
    )
    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": "moonshotai/Kimi-K2-Instruct:novita"
    }

    try:
        hf_payload = query(payload)
        if hf_payload is None:
             return {
                "language": "NULL", "message": "Failed", "country_code": "NULL", "source": "fallback-missing-key"
            }
             
        # Try to extract the generated text
        content = None
        if isinstance(hf_payload, dict) and "choices" in hf_payload:
            choices = hf_payload["choices"]
            if isinstance(choices, list) and choices and "message" in choices[0] and "content" in choices[0]["message"]:
                content = choices[0]["message"]["content"]
        
        response_obj = None
        if content:
            # FIX 2: More robust JSON extraction/cleaning
            # Look for the last valid JSON object in the response
            json_str = _re.search(r'\{[\s\S]*?\}', content)
            if json_str:
                 content = json_str.group(0)
            
            try:
                response_obj = _json.loads(content)
            except Exception as e:
                print(f"Could not parse instruct JSON output: '{content}' :: {e}")
                
        if (
            response_obj
            and isinstance(response_obj, dict)
            and "message" in response_obj
            and "language" in response_obj
            and "country_code" in response_obj
        ):
            # Store in cache for similar IPs
            _add_to_cache(
                ip,
                response_obj["country_code"].strip().upper(),
                response_obj["language"].strip().lower(),
                response_obj["message"].strip(),
            )
            return {
                "language": response_obj["language"].strip().lower(),
                "message": response_obj["message"].strip(),
                "country_code": response_obj["country_code"].strip().upper(),
                "source": "ai"
            }
        else:
            print(f"Response did not contain valid keys or valid JSON structure: {hf_payload}")
    except Exception as e:
        print(f"Hugging Face instruct API error: {e}")

    return {
        "language": "NULL",
        "message": "Failed",
        "country_code": "NULL",
        "source": "fallback-api-error"
    }


def get_welcome_message(ip: str) -> Dict[str, Any]:
    """
    Uses an LLM to:
      1. Infer country code (from IP)
      2. Infer language code and localized welcome message (single step)
    Supports caching to avoid repeated LLM lookups for similar IPs.
    """
    result = instruct_get_localized_welcome_from_ip(ip)
    return {
        "message": result["message"],
        "language": result["language"],
        "country_code": result["country_code"],
        "ip_used": ip,
        "source": result["source"]
    }