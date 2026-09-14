"""
echoX - Audio Preprocessor & DSP Pipeline
Standardizes incoming audio via:
1. Resampling to 16 kHz & Mono conversion.
2. Direct Slicing / Zero-padding to 64,600 samples (AASIST window).
3. Streaming buffer management.
"""

import io
import numpy as np
import soundfile as sf
from scipy import signal
from typing import Tuple, Optional, Dict, Any
from backend.app.core.config import AUDIO_CONFIG


class AudioProcessor:
    """
    Standardizes audio inputs for AASIST inference using direct slicing and zero-padding.
    """

    def __init__(
        self,
        target_sample_rate: int = AUDIO_CONFIG.SAMPLE_RATE,
        target_samples: int = AUDIO_CONFIG.SAMPLE_WINDOW
    ):
        self.target_sample_rate = target_sample_rate
        self.target_samples = target_samples

    def load_audio_from_bytes(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        with io.BytesIO(audio_bytes) as bio:
            data, sr = sf.read(bio, dtype="float32")
        return data, sr

    def to_mono(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim > 1:
            return np.mean(audio, axis=1)
        return audio

    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr == self.target_sample_rate:
            return audio.astype(np.float32)
        target_length = int(round(len(audio) * float(self.target_sample_rate) / orig_sr))
        resampled = signal.resample(audio, target_length)
        return resampled.astype(np.float32)

    def enforce_aasist_window(self, audio: np.ndarray) -> np.ndarray:
        """
        Direct Slicing / Zero-padding to enforce exactly 64,600 samples.
        - If shorter: zero-pad directly to 64,600 samples.
        - If longer: slice directly to first 64,600 samples.
        """
        length = len(audio)
        if length == self.target_samples:
            return audio.astype(np.float32)

        if length < self.target_samples:
            padded = np.pad(audio, (0, self.target_samples - length), mode="constant")
            return padded.astype(np.float32)
        else:
            sliced = audio[:self.target_samples]
            return sliced.astype(np.float32)

    def process_file_bytes(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Ingestion pipeline:
        1. Decode bytes -> (data, sr)
        2. Mono conversion
        3. Resampling to 16 kHz
        4. Direct slicing or zero-padding to 64,600 samples
        """
        data, sr = self.load_audio_from_bytes(audio_bytes)
        mono = self.to_mono(data)
        resampled = self.resample(mono, sr)
        windowed = self.enforce_aasist_window(resampled)

        return {
            "processed_audio": windowed,
            "duration_seconds": len(resampled) / self.target_sample_rate,
            "sample_rate": self.target_sample_rate
        }


class StreamBuffer:
    """
    Buffer for streaming audio chunks. Accumulates samples and extracts 64,600-sample windows.
    """

    def __init__(
        self,
        target_samples: int = AUDIO_CONFIG.SAMPLE_WINDOW,
        hop_samples: int = AUDIO_CONFIG.HOP_LENGTH
    ):
        self.target_samples = target_samples
        self.hop_samples = hop_samples
        self.buffer = np.zeros(0, dtype=np.float32)

    def append_chunk(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        self.buffer = np.concatenate([self.buffer, chunk])
        if len(self.buffer) >= self.target_samples:
            window = self.buffer[:self.target_samples].copy()
            self.buffer = self.buffer[self.hop_samples:]
            return window
        return None

    def reset(self):
        self.buffer = np.zeros(0, dtype=np.float32)


AUDIO_PROCESSOR = AudioProcessor()
