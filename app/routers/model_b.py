from fastapi import APIRouter
from app.schemas.model_b import ModelBInput, ModelBOutput
from app.services.model_b_service import predict

router = APIRouter(prefix="/model-b", tags=["Model B"])

@router.post("/predict", response_model=ModelBOutput)
def predict_model_b(payload: ModelBInput):
    return predict(payload.text)
