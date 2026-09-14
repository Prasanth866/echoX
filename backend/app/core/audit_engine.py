"""
echoX - Backend Audit Compliance Logger
Maintains an internal backend audit trail for voice security verification events.
"""

import hashlib
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from backend.app.core.config import AUDIT_CONFIG


class AuditRecord(BaseModel):
    log_id: int
    timestamp: float
    iso_timestamp: str
    audio_sha256: str
    risk_score: float
    verdict: str
    action: str
    filename: Optional[str] = None


class AuditLogger:
    """
    Backend audit compliance logger.
    """

    def __init__(self):
        self.logs: List[AuditRecord] = []

    @staticmethod
    def compute_sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def log_event(
        self,
        audio_bytes: bytes,
        risk_score: float,
        verdict: str,
        action: str,
        filename: Optional[str] = None
    ) -> AuditRecord:
        now = time.time()
        iso_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
        audio_hash = self.compute_sha256(audio_bytes)

        record = AuditRecord(
            log_id=len(self.logs) + 1,
            timestamp=now,
            iso_timestamp=iso_time,
            audio_sha256=audio_hash,
            risk_score=round(risk_score, 2),
            verdict=verdict,
            action=action,
            filename=filename
        )
        self.logs.append(record)
        if len(self.logs) > AUDIT_CONFIG.MAX_AUDIT_LOGS:
            self.logs.pop(0)

        return record

    def get_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        records = self.logs[-limit:]
        return [r.model_dump() for r in reversed(records)]


AUDIT_LOGGER = AuditLogger()
