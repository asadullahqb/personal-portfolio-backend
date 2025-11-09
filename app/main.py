from fastapi import FastAPI
from app.routers import health, welcome

app = FastAPI(title="Multi-Model ML API")

app.include_router(health.router)
app.include_router(welcome.router)
