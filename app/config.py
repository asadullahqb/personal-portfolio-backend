from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_a_path: str = "app/models/model_a.pkl"
    model_b_path: str = "app/models/model_b"  # directory for transformer
    mlflow_tracking_uri: str = "file:///tmp/mlruns"
    country_to_language: str = "t5-small"
    translator_model_name: str = "Helsinki-NLP/opus-mt-en-mul"

settings = Settings()
