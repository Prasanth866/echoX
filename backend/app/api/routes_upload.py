"""
echoX - REST File Upload & Analysis API
Accepts audio file uploads, runs inference, and records backend audit compliance logs.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any

from backend.app.services.audio_processor import AUDIO_PROCESSOR
from backend.app.services.detector import DETECTOR_SERVICE
from backend.app.core.audit_engine import AUDIT_LOGGER

router = APIRouter(prefix="/api/v1", tags=["Audio Inspection"])


@router.post("/analyze-file")
async def analyze_audio_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyzes an uploaded audio file (.wav, .mp3, .flac).
    Performs direct slicing/zero-padding, AASIST feature extraction, 0-100 risk scoring,
    and logs the verification in backend audit history.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        proc_result = AUDIO_PROCESSOR.process_file_bytes(content)
        det_result = DETECTOR_SERVICE.detect(proc_result["processed_audio"])

        AUDIT_LOGGER.log_event(
            audio_bytes=content,
            risk_score=det_result["risk_score"],
            verdict=det_result["verdict"],
            action=det_result["action"],
            filename=file.filename
        )

        return {
            "filename": file.filename,
            "duration_seconds": proc_result["duration_seconds"],
            **det_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/audit-logs")
async def get_audit_logs(limit: int = 20) -> Dict[str, Any]:
    """
    Backend compliance audit log query.
    """
    logs = AUDIT_LOGGER.get_logs(limit=limit)
    return {
        "total_logs": len(AUDIT_LOGGER.logs),
        "logs": logs
    }
