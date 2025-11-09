from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app

client = TestClient(app)

def test_predict_model_a():
    response = client.post("/model-a/predict", json={"features": [1,2,3,4]})
    assert response.status_code == 200
    assert "prediction" in response.json()
