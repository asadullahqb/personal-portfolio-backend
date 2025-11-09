from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    mlflow_tracking_uri: str = "file:///tmp/mlruns"
    country_to_language: str = "t5-small"
    translator_model_name: str = "Helsinki-NLP/opus-mt-en-mul"

settings = Settings()
