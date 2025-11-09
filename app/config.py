from pydantic import BaseSettings

class Settings(BaseSettings):
    model_a_path: str = "app/models/model_a.pkl"
    model_b_path: str = "app/models/model_b"  # directory for transformer
    mlflow_tracking_uri: str = "file:///tmp/mlruns"
    translator_model_name: str = "Helsinki-NLP/opus-mt-en-mul"

settings = Settings()
