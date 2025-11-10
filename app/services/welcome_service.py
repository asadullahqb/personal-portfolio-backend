import os
import requests
from typing import Dict, Any
from collections import defaultdict
import threading
import ipaddress

current_env = os.environ.get('VERCEL_ENV', 'development')

# Load environment variables from a .env file ONLY if running in development mode.
if current_env == 'development':
    # This assumes your local .env file is in the root directory
    from dotenv import load_dotenv
    load_dotenv()
    print("Environment: Local Development. Loaded variables from .env file.")
else:
    # In Vercel (production or preview), the variables are securely injected 
    # via the Vercel dashboard and do not need a .env file.
    print(f"Environment: {current_env.capitalize()}. Using Vercel environment variables.")

API_URL = "https://router.huggingface.co/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {os.environ['HF_API_KEY']}",
}

# Thread-safe in-memory cache: {ip_prefix (first 3 octets): {country_code: {message, language}}}
_ip_cache = {}
_cache_lock = threading.Lock()


def _ip_prefix(ip: str, prefix_octets: int = 3) -> str:
    """Return the first N octets of the IPv4 address as a prefix string."""
    try:
        parts = ip.split('.')
        if len(parts) != 4:
            raise ValueError("Invalid IP format")
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

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


def instruct_get_localized_welcome_from_ip(ip: str) -> Dict[str, str]:
    """
    Uses an LLM (via Hugging Face API) to determine:
      - Country (2-letter ISO code)
      - Primary language (ISO 639-1)
      - The relevant localized welcome message
    Accepts a public IPv4 address as input.
    Utilizes a cache to avoid unnecessary LLM queries for similar IPs.
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
        # Try to extract the generated text, which may be inside choices[0]['message']['content']
        content = None
        if isinstance(hf_payload, dict) and "choices" in hf_payload:
            choices = hf_payload["choices"]
            if isinstance(choices, list) and choices and "message" in choices[0] and "content" in choices[0]["message"]:
                content = choices[0]["message"]["content"]
        import json as _json
        import re as _re
        response_obj = None
        if content:
            match = _re.search(r"\{[\s\S]*?\}", content)
            if match:
                content = match.group(0)
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
            print(f"Response did not contain valid keys: {hf_payload}")
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