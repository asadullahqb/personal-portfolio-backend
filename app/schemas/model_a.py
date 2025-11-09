from pydantic import BaseModel
from typing import List

class ModelAInput(BaseModel):
    features: List[float]

class ModelAOutput(BaseModel):
    prediction: int
