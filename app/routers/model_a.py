from fastapi import APIRouter
from app.schemas.model_a import ModelAInput, ModelAOutput
from app.services.model_a_service import predict

router = APIRouter(prefix="/model-a", tags=["Model A"])

@router.post("/predict", response_model=ModelAOutput)
def predict_model_a(payload: ModelAInput):
    return predict(payload.features)
