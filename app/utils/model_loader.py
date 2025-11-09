from functools import lru_cache
import joblib
import torch
from app.config import settings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

@lru_cache()
def load_translator_model():
    tokenizer = AutoTokenizer.from_pretrained(settings.translator_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(settings.translator_model_name)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model
