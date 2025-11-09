from fastapi import APIRouter
from pydantic import BaseModel
from app.services.welcome_service import get_welcome_message

router = APIRouter(prefix="/welcome", tags=["Welcome Translator"])

class WelcomeInput(BaseModel):
    ip: str

class WelcomeOutput(BaseModel):
    message: str

@router.post("/", response_model=WelcomeOutput)
def welcome(payload: WelcomeInput):
    """Return AI-generated Welcome message for given IP."""
    return get_welcome_message(payload.ip)
