from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    country_to_language: str = "t5-small"
    translator_model_name: str = "Helsinki-NLP/opus-mt-en-mul"

settings = Settings()
