"""
echoX - Synthetic Audio Sample Generator
Generates realistic 16 kHz calibration WAV samples:
1. genuine_human.wav: Simulates natural human speech with glottal pulses and formant harmonics.
2. cloned_elevenlabs.wav: Simulates AI vocoder synthesis with high-frequency phase artifacts.
"""

import numpy as np
import soundfile as sf
from pathlib import Path

SAMPLE_RATE = 16000
DURATION = 4.1
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
t = np.linspace(0, DURATION, NUM_SAMPLES, endpoint=False)

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_genuine_human():
    """Generates acoustic pattern characteristic of human vocal tract (formants F1, F2, F3)."""
    f0 = 125.0 + 5.0 * np.sin(2 * np.pi * 3.5 * t) + np.random.normal(0, 0.5, NUM_SAMPLES)
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE

    signal = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase) + 0.1 * np.sin(4 * phase)

    f1 = np.sin(2 * np.pi * 600 * t) * 0.4
    f2 = np.sin(2 * np.pi * 1500 * t) * 0.2
    f3 = np.sin(2 * np.pi * 2500 * t) * 0.1
    signal = signal + f1 + f2 + f3

    envelope = np.maximum(0, np.sin(2 * np.pi * 1.8 * t)) ** 1.5
    signal = signal * envelope

    signal = signal / (np.max(np.abs(signal)) + 1e-6) * 0.75
    return signal.astype(np.float32)


def generate_cloned_elevenlabs():
    """Generates acoustic pattern characteristic of neural vocoders with high-frequency artifacts."""
    f0 = 160.0
    phase = 2 * np.pi * f0 * t

    signal = (
        np.sin(phase)
        + 0.8 * np.sin(2 * phase)
        + 0.6 * np.sin(3 * phase)
        + 0.5 * np.sin(4 * phase)
        + 0.45 * np.sin(5 * phase)
    )

    vocoder_ripple = 0.35 * np.sin(2 * np.pi * 5200 * t) + 0.25 * np.sin(2 * np.pi * 6800 * t)
    noise_hash = np.random.normal(0, 0.08, NUM_SAMPLES)
    signal = signal + vocoder_ripple + noise_hash

    envelope = np.clip(np.sin(2 * np.pi * 2.0 * t) * 2.0, 0.1, 1.0)
    signal = signal * envelope

    signal = signal / (np.max(np.abs(signal)) + 1e-6) * 0.85
    return signal.astype(np.float32)


def main():
    human = generate_genuine_human()
    sf.write(OUT_DIR / "genuine_human.wav", human, SAMPLE_RATE)
    print(f"Generated {OUT_DIR / 'genuine_human.wav'}")

    cloned = generate_cloned_elevenlabs()
    sf.write(OUT_DIR / "cloned_elevenlabs.wav", cloned, SAMPLE_RATE)
    print(f"Generated {OUT_DIR / 'cloned_elevenlabs.wav'}")


if __name__ == "__main__":
    main()
