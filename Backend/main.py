from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from app.routers import inspect, meta

app = FastAPI()

app.include_router(inspect.router, prefix="/api/v1", tags=["inspect"])
app.include_router(meta.router, prefix="/api/v1", tags=["meta"])

app.mount("/samples", StaticFiles(directory="samples"), name="samples")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Terjadi kesalahan tak terduga di server. Silakan coba lagi."},
    )


@app.get("/healthz")
def health_check():
    return {"status": "ok"}