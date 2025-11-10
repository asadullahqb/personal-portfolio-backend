import os
import logging
logging.basicConfig(level=logging.INFO)
import dotenv
dotenv.load_dotenv()
logging.info(f"Vercel Startup check. HF_API_KEY status: {'SET' if os.environ.get('HF_API_KEY') else 'MISSING'}")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import health, welcome

app = FastAPI(title="Multi-Model ML API")

# Add CORS middleware directly after creating the app instance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],  # Allow local frontend and wildcard (Vercel, etc.)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all standard HTTP methods
    allow_headers=["*"],  # Allow all standard headers (including Authorization, Content-Type)
)

app.include_router(health.router)
app.include_router(welcome.router)
