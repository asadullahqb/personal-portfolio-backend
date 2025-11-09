from functools import lru_cache
import joblib
# Assuming 'settings' is defined and accessible via your app's config system
from app.config import settings 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional, Tuple

@lru_cache
def load_translator_model() -> Tuple[Optional[AutoTokenizer], Optional[AutoModelForSeq2SeqLM]]:
    """Loads the tokenizer and model for the primary translation task (Opus-MT)."""
    model_name = settings.translator_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # CRITICAL FIX 1: Ensure model is moved to device before being cached
    if model.device != DEVICE:
        model.to(DEVICE)
        
    return tokenizer, model

@lru_cache
def load_global_language_inferencer() -> Tuple[Optional[AutoTokenizer], Optional[AutoModelForSeq2SeqLM]]:
    """Loads the tokenizer and model for the Country Code -> Language Code inference (T5-small)."""
    model_name = settings.country_to_language
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # CRITICAL FIX 2: Ensure model is moved to device before being cached
    if model.device != DEVICE:
        model.to(DEVICE)

    return tokenizer, model