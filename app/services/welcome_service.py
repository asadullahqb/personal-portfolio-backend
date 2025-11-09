import os
import requests
from typing import Dict, Any

import dotenv
dotenv.load_dotenv()

API_URL = "https://router.huggingface.co/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {os.environ['HF_API_KEY']}",
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
    """
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
    """
    result = instruct_get_localized_welcome_from_ip(ip)
    return {
        "message": result["message"],
        "language": result["language"],
        "country_code": result["country_code"],
        "ip_used": ip,
        "source": result["source"]
    }