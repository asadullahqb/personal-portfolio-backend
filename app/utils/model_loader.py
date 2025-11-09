from functools import lru_cache
import joblib
import torch
from app.config import settings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional, Tuple

@lru_cache
def load_model_a():
    return joblib.load(settings.model_a_path)

@lru_cache
def load_model_b():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_name = settings.model_b_path  # path to small transformer dir
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

@lru_cache
def load_translator_model() -> Tuple[Optional[AutoTokenizer], Optional[AutoModelForSeq2SeqLM]]:
    """Loads the tokenizer and model for the primary translation task (Opus-MT)."""
    model_name = settings.translator_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

@lru_cache
def load_global_language_inferencer() -> Tuple[Optional[AutoTokenizer], Optional[AutoModelForSeq2SeqLM]]:
    """Loads the tokenizer and model for the Country Code -> Language Code inference (T5-small)."""
    model_name = settings.country_to_language
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model
