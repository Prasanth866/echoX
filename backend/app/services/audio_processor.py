"""
echoX - Audio Preprocessor & DSP Pipeline
Standardizes incoming audio via:
1. Resampling to 16 kHz & Mono conversion.
2. Direct Slicing / Zero-padding to standard 4.0-second window (64,000 samples).
3. Streaming buffer management.
"""

import io
import numpy as np
import soundfile as sf
import torch
import librosa
from scipy import signal
from typing import Tuple, Optional, Dict, Any
from backend.app.core.config import AUDIO_CONFIG


def clean_and_normalize_audio(audio_data: np.ndarray, sr: int = 16000) -> Tuple[Optional[torch.Tensor], bool]:
    """
    VAD & Amplitude Normalization for Facebook Wav2Vec 2.0 Ingestion:
    1. Convert to mono
    2. Resample to 16 kHz
    3. Trim Leading/Trailing Silence (VAD)
    4. Check if there is actual speech (Energy Gate)
    5. Peak & RMS Normalization
    6. Fixed 4.0-second Window Padding (64,000 samples) with wrap/constant mode
    """
    if audio_data.ndim > 1:
        if audio_data.shape[0] < audio_data.shape[1]:
            audio_data = np.mean(audio_data, axis=0)
        else:
            audio_data = np.mean(audio_data, axis=1)

    if sr != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000, res_type="soxr_hq")

    trimmed_audio, _ = librosa.effects.trim(audio_data, top_db=25)

    energy = float(np.mean(trimmed_audio ** 2))
    if energy < 1e-4 or len(trimmed_audio) < 8000:
        return None, False

    trimmed_audio = trimmed_audio / (np.max(np.abs(trimmed_audio)) + 1e-6)

    target_samples = AUDIO_CONFIG.SAMPLE_WINDOW
    if len(trimmed_audio) < target_samples:
        padded = np.pad(trimmed_audio, (0, target_samples - len(trimmed_audio)), mode="constant")
    else:
        padded = trimmed_audio[:target_samples]

    return torch.FloatTensor(padded).unsqueeze(0), True


class AudioProcessor:
    """
    Standardizes audio inputs for Facebook Wav2Vec 2.0 inference using direct slicing and zero-padding.
    """

    def __init__(
        self,
        target_sample_rate: int = AUDIO_CONFIG.SAMPLE_RATE,
        target_samples: int = AUDIO_CONFIG.SAMPLE_WINDOW
    ):
        self.target_sample_rate = target_sample_rate
        self.target_samples = target_samples

    def load_audio_from_bytes(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        try:
            with io.BytesIO(audio_bytes) as bio:
                data, sr = sf.read(bio, dtype="float32")
            return data, sr
        except Exception:
            import subprocess
            try:
                proc = subprocess.run([
                    "ffmpeg", "-y", "-i", "pipe:0", "-f", "wav", "-acodec", "pcm_s16le", "-ar", str(self.target_sample_rate), "-ac", "1", "pipe:1"
                ], input=audio_bytes, capture_output=True)
                if proc.returncode == 0 and len(proc.stdout) > 0:
                    data, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
                    return data, sr
            except Exception:
                pass
            from pydub import AudioSegment
            with io.BytesIO(audio_bytes) as bio:
                seg = AudioSegment.from_file(bio)
                seg = seg.set_frame_rate(self.target_sample_rate).set_channels(1).set_sample_width(2)
                data = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
                return data, seg.frame_rate

    def to_mono(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim > 1:
            if audio.shape[0] < audio.shape[1]:
                return np.mean(audio, axis=0)
            return np.mean(audio, axis=1)
        return audio

    def resample(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr == self.target_sample_rate:
            return audio.astype(np.float32)
        try:
            return librosa.resample(
                audio.astype(np.float32),
                orig_sr=orig_sr,
                target_sr=self.target_sample_rate,
                res_type="soxr_hq"
            ).astype(np.float32)
        except Exception:
            target_length = int(round(len(audio) * float(self.target_sample_rate) / orig_sr))
            return signal.resample(audio, target_length).astype(np.float32)

    def enforce_audio_window(self, audio: np.ndarray) -> np.ndarray:
        """
        Direct Slicing / Zero-padding to enforce exactly 4.0-second (64,000 samples) window.
        - If shorter: zero-pad directly to target_samples.
        - If longer: slice directly to first target_samples.
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

    enforce_aasist_window = enforce_audio_window

    def process_file_bytes(self, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Ingestion pipeline:
        1. Decode bytes -> (data, sr)
        2. Mono conversion
        3. Resampling to 16 kHz
        4. Direct slicing or zero-padding to 4.0 seconds (64,000 samples)
        """
        data, sr = self.load_audio_from_bytes(audio_bytes)
        mono = self.to_mono(data)
        resampled = self.resample(mono, sr)
        windowed = self.enforce_audio_window(resampled)

        return {
            "processed_audio": windowed,
            "duration_seconds": len(resampled) / self.target_sample_rate,
            "sample_rate": self.target_sample_rate
        }


class StreamBuffer:
    """
    Buffer for streaming audio chunks. Accumulates samples and extracts 64,000-sample (4.0s) windows.
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
