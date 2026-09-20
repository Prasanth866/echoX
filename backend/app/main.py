"""
echoX - FastAPI Application Entrypoint
Multi-Tier Deepfake Voice & Audio Anti-Spoofing Detection Platform
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.app.api.routes_upload import router as upload_router
from backend.app.api.routes_stream import router as stream_router
from backend.app.services.detector import DETECTOR_SERVICE


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("echoX Server Initialized")
    print(f"Device: {DETECTOR_SERVICE.device}")
    print("=" * 60)
    yield
    print("echoX Server Shutting Down")


app = FastAPI(
    title="echoX Engine",
    description="Multi-Tier Deepfake Voice & Anti-Spoofing Detection Platform",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(stream_router)


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from backend.app.core.config import TEST_SAMPLES_DIR, BASE_DIR

dashboard_dir = BASE_DIR / "frontend" / "web_dashboard"
if dashboard_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")

if TEST_SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=str(TEST_SAMPLES_DIR)), name="samples")


@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "service": "echoX",
        "model": "Facebook Wav2Vec 2.0",
        "device": str(DETECTOR_SERVICE.device),
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    index_file = dashboard_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "message": "Welcome to echoX Anti-Spoofing Core Engine (Facebook Wav2Vec 2.0)",
        "docs": "/docs",
        "health": "/health",
        "dashboard": "/dashboard"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
