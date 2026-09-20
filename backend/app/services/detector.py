"""
echoX - Detection Service & Facebook Wav2Vec 2.0 Inference Engine
Manages:
1. Facebook Wav2Vec 2.0 neural backbone loading & device placement (MPS / CUDA / CPU).
2. Energy gating & acoustic silence filtering.
3. Wav2Vec 2.0 representation inference -> Risk Engine policy evaluation.
"""

import time
from typing import Dict, Any, Optional
import numpy as np
import torch

from backend.app.core.config import AUDIO_CONFIG, MODEL_CONFIG
from backend.app.core.risk_engine import RISK_ENGINE
from backend.app.models.wav2vec import FacebookWav2Vec2Detector


class DetectorService:
    """
    Core inference engine powered by Facebook Wav2Vec 2.0 for voice deepfake detection.
    """

    def __init__(self):
        self.device = self._select_device()
        self.model: Optional[FacebookWav2Vec2Detector] = None
        self._load_model()

    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_model(self):
        print(f"[Detector] Initializing Facebook Wav2Vec 2.0 engine on device: {self.device}")
        self.model = FacebookWav2Vec2Detector(device=self.device)

    def _check_silence_or_ambient(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Energy and RMS check to filter out pure flatline silence.
        """
        peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
        rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) > 0 else 0.0

        if peak < 1e-4 or rms < 1e-5:
            return {
                "is_silence": True,
                "rms": round(rms, 6),
                "peak": round(peak, 6),
                "vocal_ratio": 0.0
            }

        return {
            "is_silence": False,
            "rms": round(rms, 6),
            "peak": round(peak, 6),
            "vocal_ratio": 1.0
        }

    def detect(self, audio_window: np.ndarray) -> Dict[str, Any]:
        """
        Runs deepfake voice inference on audio array (4.0s / 64,000 samples @ 16kHz)
        using Facebook Wav2Vec 2.0 neural backbone.
        """
        start_time = time.perf_counter()

        heuristics = self._check_silence_or_ambient(audio_window)
        if heuristics.get("is_silence", False):
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            return {
                "verdict": "INACTIVE",
                "action": "ALLOW",
                "risk_score": 0.0,
                "spoof_probability": 0.0,
                "bonafide_probability": 1.0,
                "color": "#64748B",
                "description": "Audio is silent or empty. Please provide an audible speech recording.",
                "inference_time_ms": latency_ms,
                "acoustic_features": heuristics,
                "model_name": MODEL_CONFIG.MODEL_NAME
            }

        spoof_prob, bonafide_prob, latents_meta = self.model(
            audio_window,
            sr=AUDIO_CONFIG.SAMPLE_RATE
        )

        risk_score = RISK_ENGINE.compute_risk_score(spoof_prob)
        decision = RISK_ENGINE.evaluate_decision(risk_score)
        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        return {
            **decision,
            "spoof_probability": round(float(spoof_prob), 4),
            "bonafide_probability": round(float(bonafide_prob), 4),
            "inference_time_ms": latency_ms,
            "model_name": MODEL_CONFIG.MODEL_NAME,
            "latents_meta": latents_meta,
            "acoustic_features": heuristics
        }


DETECTOR_SERVICE = DetectorService()
