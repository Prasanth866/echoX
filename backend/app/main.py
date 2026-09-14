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


@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "service": "echoX",
        "device": str(DETECTOR_SERVICE.device),
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    return {
        "message": "Welcome to echoX Anti-Spoofing Core Engine",
        "docs": "/docs",
        "health": "/health",
        "api_endpoints": {
            "file_analysis": "/api/v1/analyze-file",
            "audit_logs": "/api/v1/audit-logs",
            "websocket_stream": "/ws/stream"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
