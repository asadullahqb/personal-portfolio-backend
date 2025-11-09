from pydantic import BaseModel

class ModelBInput(BaseModel):
    text: str

class ModelBOutput(BaseModel):
    label: str
