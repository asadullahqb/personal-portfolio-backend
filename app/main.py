import os

current_env = os.environ.get('VERCEL_ENV', 'development')

# Load environment variables from a .env file ONLY if running in development mode.
if current_env == 'development':
    # This assumes your local .env file is in the root directory
    from dotenv import load_dotenv
    load_dotenv()
    print("Environment: Local Development. Loaded variables from .env file.")
else:
    # In Vercel (production or preview), the variables are securely injected 
    # via the Vercel dashboard and do not need a .env file.
    print(f"Environment: {current_env.capitalize()}. Using Vercel environment variables.")

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
