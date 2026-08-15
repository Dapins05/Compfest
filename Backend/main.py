from fastapi import FastAPI
from app.routers import inspect

app = FastAPI()

app.include_router(inspect.router, prefix="/api/v1", tags=["inspect"])


@app.get("/healthz")
def health_check():
    return {"status": "ok"}