from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_model_a():
    response = client.post("/model-a/predict", json={"features": [1,2,3,4]})
    assert response.status_code == 200
    assert "prediction" in response.json()
