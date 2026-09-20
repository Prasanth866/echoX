"""
echoX - Core Configuration
Defines risk score thresholds, audio processing parameters, and model configurations.
"""

from pathlib import Path
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
WEIGHTS_DIR = BACKEND_DIR / "weights"
TEST_SAMPLES_DIR = BASE_DIR / "tests" / "samples"

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
TEST_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


class RiskThresholdConfig(BaseModel):
    """
    Risk Score Thresholds (0 - 100):
    - 0  to 30: Low Risk (Bona fide human) -> ALLOW
    - 31 to 60: Medium Risk (Indeterminate artifacts) -> CHALLENGE_MFA
    - 61 to 100: High Risk (Spoofed / Cloned voice) -> BLOCK_ALERT
    """
    LOW_MAX: float = Field(default=30.0, description="Upper bound for Low Risk")
    MED_MAX: float = Field(default=60.0, description="Upper bound for Medium Risk")
    HIGH_MAX: float = Field(default=100.0, description="Upper bound for High Risk")


class AudioConfig(BaseModel):
    """
    Audio DSP & Standard 4.0-second Window Parameters (16 kHz)
    """
    SAMPLE_RATE: int = Field(default=16000, description="Standardized sampling rate (16 kHz)")
    SAMPLE_WINDOW: int = Field(default=64000, description="Standard 4.0-second window (64,000 samples @ 16 kHz)")
    HOP_LENGTH: int = Field(default=16000, description="Sliding window hop size (16,000 samples / 1.0s)")
    CHUNK_SIZE: int = Field(default=4000, description="Streaming chunk size (4,000 samples / 0.25s)")


class ModelConfig(BaseModel):
    """
    Model architecture & checkpoint settings: Facebook Wav2Vec 2.0
    """
    MODEL_NAME: str = "Facebook Wav2Vec 2.0"
    PRIMARY_MODEL: str = "facebook/wav2vec2-base"
    FINE_TUNED_MODEL: str = "MelodyMachine/Deepfake-audio-detection-V2"
    WEIGHTS_PATH: Path = WEIGHTS_DIR / "wav2vec2_classifier.pth"
    SPOOF_CLASS_INDEX: int = Field(default=0, description="Index for spoof logit probability")


class AuditConfig(BaseModel):
    """
    Backend audit compliance logging settings
    """
    AUDIT_ENABLED: bool = True
    MAX_AUDIT_LOGS: int = 500


RISK_CONFIG = RiskThresholdConfig()
AUDIO_CONFIG = AudioConfig()
MODEL_CONFIG = ModelConfig()
AUDIT_CONFIG = AuditConfig()
