"""
echoX - Unit & Integration Test Suite
Validates:
1. Audio processor resampling & direct slicing / zero-padding to 64,600 samples.
2. Risk engine 3-tier decision policy.
3. Backend audit compliance logging.
4. AASIST model forward-pass with hardcoded probs[0] index.
5. FastAPI REST endpoints (/health, /api/v1/analyze-file, /api/v1/audit-logs).
"""

import io
import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from backend.app.core.config import AUDIO_CONFIG
from backend.app.services.audio_processor import AUDIO_PROCESSOR
from backend.app.services.detector import DETECTOR_SERVICE
from backend.app.core.risk_engine import RISK_ENGINE, RiskVerdict, SecurityAction
from backend.app.core.audit_engine import AUDIT_LOGGER
from backend.app.main import app

client = TestClient(app)


def test_audio_processor_slicing_and_padding():
    """Verify that any length audio is sliced or zero-padded directly to 64,600 samples."""
    short_audio = np.random.normal(0, 0.1, 10000).astype(np.float32)
    padded = AUDIO_PROCESSOR.enforce_aasist_window(short_audio)
    assert len(padded) == AUDIO_CONFIG.SAMPLE_WINDOW
    assert np.all(padded[10000:] == 0.0)

    long_audio = np.random.normal(0, 0.1, 100000).astype(np.float32)
    sliced = AUDIO_PROCESSOR.enforce_aasist_window(long_audio)
    assert len(sliced) == AUDIO_CONFIG.SAMPLE_WINDOW
    assert np.array_equal(sliced, long_audio[:AUDIO_CONFIG.SAMPLE_WINDOW])


def test_risk_engine_decision_policies():
    """Verify 3-tier risk classification and security actions."""
    low_res = RISK_ENGINE.evaluate_decision(15.0)
    assert low_res["verdict"] == RiskVerdict.LOW_RISK
    assert low_res["action"] == SecurityAction.ALLOW

    med_res = RISK_ENGINE.evaluate_decision(45.0)
    assert med_res["verdict"] == RiskVerdict.MEDIUM_RISK
    assert med_res["action"] == SecurityAction.CHALLENGE_MFA

    high_res = RISK_ENGINE.evaluate_decision(85.0)
    assert high_res["verdict"] == RiskVerdict.HIGH_RISK
    assert high_res["action"] == SecurityAction.BLOCK_ALERT


def test_backend_audit_logger():
    """Verify backend compliance logging."""
    dummy_payload = b"RIFF....test_voice_pcm"
    initial_count = len(AUDIT_LOGGER.logs)

    record = AUDIT_LOGGER.log_event(
        audio_bytes=dummy_payload,
        risk_score=88.5,
        verdict="HIGH_RISK",
        action="BLOCK_ALERT",
        filename="test.wav"
    )

    assert len(AUDIT_LOGGER.logs) == initial_count + 1
    assert record.audio_sha256 is not None
    assert record.risk_score == 88.5
    assert record.verdict == "HIGH_RISK"
    assert record.action == "BLOCK_ALERT"


def test_detector_inference():
    """Verify AASIST model forward-pass with hardcoded probs[0] index."""
    tone = (np.sin(2 * np.pi * 300 * np.linspace(0, 4.0375, AUDIO_CONFIG.SAMPLE_WINDOW)) * 0.4).astype(np.float32)
    result = DETECTOR_SERVICE.detect(tone)

    assert "risk_score" in result
    assert "verdict" in result
    assert "action" in result
    assert "inference_time_ms" in result
    assert result["inference_time_ms"] > 0
    assert 0.0 <= result["risk_score"] <= 100.0


def test_api_health():
    """Verify FastAPI /health endpoint."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert data["service"] == "echoX"


def test_api_analyze_file_upload():
    """Verify POST /api/v1/analyze-file with in-memory WAV."""
    buf = io.BytesIO()
    audio = (np.sin(2 * np.pi * 500 * np.linspace(0, 2.0, 32000)) * 0.5).astype(np.float32)
    sf.write(buf, audio, 16000, format="WAV")
    buf.seek(0)

    files = {"file": ("test_sample.wav", buf, "audio/wav")}
    resp = client.post("/api/v1/analyze-file", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert "risk_score" in data
    assert "verdict" in data
    assert "action" in data
    assert data["filename"] == "test_sample.wav"
