from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routers import inspect, meta

app = FastAPI()

app.include_router(inspect.router, prefix="/api/v1", tags=["inspect"])
app.include_router(meta.router, prefix="/api/v1", tags=["meta"])

app.mount("/samples", StaticFiles(directory="samples"), name="samples")


@app.get("/healthz")
def health_check():
    return {"status": "ok"}