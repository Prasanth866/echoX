"""
echoX - 4-Sample Calibration Audio Generator
Generates realistic 16 kHz, 4.0-second calibration audio samples (64,000 samples):
1. sample_1_genuine_human.wav: Natural human speech with formants, micro-pitch jitter, and glottal pulses.
2. sample_2_cloned_elevenlabs.wav: Modern neural vocoder clone with high-frequency phase artifacts.
3. sample_3_neural_tts.wav: Tacotron2/WaveNet text-to-speech with harmonic spectral smoothing.
4. sample_4_voice_conversion.wav: Cross-speaker voice conversion with spectral warping.
"""

import numpy as np
import soundfile as sf
from pathlib import Path

SAMPLE_RATE = 16000
DURATION = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
t = np.linspace(0, DURATION, NUM_SAMPLES, endpoint=False)

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_genuine_human() -> np.ndarray:
    """Natural authentic speech with human glottal pulses and formant resonances (F1, F2, F3)."""
    f0 = 128.0 + 4.5 * np.sin(2 * np.pi * 3.2 * t) + np.random.normal(0, 0.35, NUM_SAMPLES)
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE

    signal = (
        np.sin(phase)
        + 0.52 * np.sin(2 * phase)
        + 0.28 * np.sin(3 * phase)
        + 0.12 * np.sin(4 * phase)
    )

    f1 = np.sin(2 * np.pi * 650 * t) * 0.38
    f2 = np.sin(2 * np.pi * 1550 * t) * 0.22
    f3 = np.sin(2 * np.pi * 2600 * t) * 0.12
    signal = signal + f1 + f2 + f3

    envelope = np.maximum(0.0, np.sin(2 * np.pi * 1.7 * t)) ** 1.8 + 0.05
    signal = signal * envelope

    signal = signal / (np.max(np.abs(signal)) + 1e-6) * 0.75
    return signal.astype(np.float32)


def generate_cloned_elevenlabs() -> np.ndarray:
    """Modern AI neural vocoder voice clone with high-frequency phase alignment and ripple."""
    f0 = 162.0 + 1.2 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE

    signal = (
        np.sin(phase)
        + 0.82 * np.sin(2 * phase)
        + 0.65 * np.sin(3 * phase)
        + 0.52 * np.sin(4 * phase)
        + 0.42 * np.sin(5 * phase)
    )

    vocoder_ripple = 0.38 * np.sin(2 * np.pi * 5400 * t) + 0.28 * np.sin(2 * np.pi * 7000 * t)
    noise_hash = np.random.normal(0, 0.07, NUM_SAMPLES)
    signal = signal + vocoder_ripple + noise_hash

    envelope = np.clip(np.sin(2 * np.pi * 2.0 * t) * 2.0, 0.15, 1.0)
    signal = signal * envelope

    signal = signal / (np.max(np.abs(signal)) + 1e-6) * 0.85
    return signal.astype(np.float32)


def generate_neural_tts() -> np.ndarray:
    """WaveNet / Tacotron TTS with rigid mechanical pitch and over-smoothed spectral formants."""
    f0 = 145.0 + 0.8 * np.sin(2 * np.pi * 0.8 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE

    signal = (
        np.sin(phase)
        + 0.70 * np.sin(2 * phase)
        + 0.45 * np.sin(3 * phase)
        + 0.30 * np.sin(4 * phase)
    )

    synth_buzz = 0.30 * np.sin(2 * np.pi * 4200 * t) + 0.18 * np.sin(2 * np.pi * 6100 * t)
    signal = signal + synth_buzz

    envelope = np.clip(np.sin(2 * np.pi * 1.5 * t) * 1.6, 0.2, 0.95)
    signal = signal * envelope

    signal = signal / (np.max(np.abs(signal)) + 1e-6) * 0.80
    return signal.astype(np.float32)


def generate_voice_conversion() -> np.ndarray:
    """Voice conversion (VC) with spectral envelope warping and boundary phase mismatch."""
    f0 = 115.0 + 2.5 * np.sin(2 * np.pi * 2.8 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE

    signal = (
        np.sin(phase)
        + 0.60 * np.sin(2 * phase)
        + 0.35 * np.sin(3 * phase)
    )

    glitches = np.zeros(NUM_SAMPLES, dtype=np.float32)
    step = int(SAMPLE_RATE * 0.125)
    for i in range(0, NUM_SAMPLES, step):
        glitches[i:min(i + 40, NUM_SAMPLES)] += np.random.normal(0, 0.35, min(40, NUM_SAMPLES - i))

    spectral_warp = 0.32 * np.sin(2 * np.pi * 4900 * t)
    signal = signal + glitches + spectral_warp

    envelope = np.clip(np.sin(2 * np.pi * 1.9 * t) * 1.7, 0.1, 0.9)
    signal = signal * envelope

    signal = signal / (np.max(np.abs(signal)) + 1e-6) * 0.80
    return signal.astype(np.float32)


def main():
    print(f"Generating 4 calibration samples (4.0s @ 16kHz = {NUM_SAMPLES} samples)...")
    
    human = generate_genuine_human()
    sf.write(OUT_DIR / "sample_1_genuine_human.wav", human, SAMPLE_RATE)
    sf.write(OUT_DIR / "genuine_human.wav", human, SAMPLE_RATE)
    print(" [1/4] sample_1_genuine_human.wav (Authentic Human Voice)")

    cloned = generate_cloned_elevenlabs()
    sf.write(OUT_DIR / "sample_2_cloned_elevenlabs.wav", cloned, SAMPLE_RATE)
    sf.write(OUT_DIR / "cloned_elevenlabs.wav", cloned, SAMPLE_RATE)
    print(" [2/4] sample_2_cloned_elevenlabs.wav (ElevenLabs AI Voice Clone)")

    tts = generate_neural_tts()
    sf.write(OUT_DIR / "sample_3_neural_tts.wav", tts, SAMPLE_RATE)
    print(" [3/4] sample_3_neural_tts.wav (WaveNet / Tacotron Neural TTS)")

    vc = generate_voice_conversion()
    sf.write(OUT_DIR / "sample_4_voice_conversion.wav", vc, SAMPLE_RATE)
    print(" [4/4] sample_4_voice_conversion.wav (Cross-Speaker Voice Conversion)")

    print(f"\nAll 4 samples generated successfully in {OUT_DIR}")


if __name__ == "__main__":
    main()
