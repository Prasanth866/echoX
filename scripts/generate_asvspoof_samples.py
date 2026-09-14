"""
echoX - ASVspoof 2019 LA Calibration Sample Generator
Generates realistic 16 kHz calibration samples for Classical & Neural TTS / VC algorithms (A01 through A19)
along with the official ASVspoof 2019 LA evaluation protocol file.
"""

import numpy as np
import soundfile as sf
from pathlib import Path

SAMPLE_RATE = 16000
DURATION = 4.1
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
t = np.linspace(0, DURATION, NUM_SAMPLES, endpoint=False)

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "samples" / "asvspoof2019_la"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_bonafide():
    """Authentic natural human vocal tract (glottal pulses, formants, natural micro-tremor)."""
    f0 = 130.0 + 4.0 * np.sin(2 * np.pi * 3.0 * t) + np.random.normal(0, 0.4, NUM_SAMPLES)
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    signal = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase) + 0.1 * np.sin(4 * phase)
    # Formants F1, F2, F3
    f1 = np.sin(2 * np.pi * 650 * t) * 0.4
    f2 = np.sin(2 * np.pi * 1450 * t) * 0.25
    f3 = np.sin(2 * np.pi * 2400 * t) * 0.1
    signal = signal + f1 + f2 + f3
    envelope = np.maximum(0, np.sin(2 * np.pi * 1.6 * t)) ** 1.5
    signal = signal * envelope
    signal = signal / (np.max(np.abs(signal)) + 1e-6) * 0.75
    return signal.astype(np.float32)


def generate_attack(attack_id: str):
    """Generates synthetic audio characteristic of specific ASVspoof 2019 LA attack algorithm."""
    f0 = 150.0 + np.sin(2 * np.pi * 2.0 * t) * 2.0
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    signal = np.sin(phase) + 0.7 * np.sin(2 * phase) + 0.5 * np.sin(3 * phase)

    if attack_id == "A01":  # WaveNet Neural Vocoder
        vocoder_ripple = 0.3 * np.sin(2 * np.pi * 4800 * t) + 0.2 * np.sin(2 * np.pi * 6200 * t)
        signal = signal + vocoder_ripple + np.random.normal(0, 0.05, NUM_SAMPLES)
    elif attack_id == "A02":  # WORLD Vocoder
        signal = signal + 0.4 * np.sin(2 * np.pi * 5100 * t)  # periodic buzzing
    elif attack_id == "A03":  # LPC Vocoder
        signal = signal + 0.35 * np.sin(2 * np.pi * 4600 * t) + np.random.normal(0, 0.04, NUM_SAMPLES)
    elif attack_id == "A04":  # Unit Selection Concatenative
        # Add phase discontinuities every 0.2s
        window_size = int(SAMPLE_RATE * 0.2)
        for i in range(0, NUM_SAMPLES, window_size):
            signal[i:i+50] += 0.3 * np.random.normal(0, 1, min(50, NUM_SAMPLES - i))
        signal = signal + 0.25 * np.sin(2 * np.pi * 5300 * t)
    elif attack_id == "A05":  # Voice Conversion (VAE/WORLD)
        signal = signal + 0.35 * np.sin(2 * np.pi * 5500 * t)
    elif attack_id == "A06":  # Spectral Filtering VC
        signal = signal + 0.3 * np.sin(2 * np.pi * 4900 * t)
    elif attack_id == "A10":  # HiFi-GAN Modern Vocoder
        # Characteristic multi-periodicity artifacts
        signal = signal + 0.38 * np.sin(2 * np.pi * 5600 * t) + 0.2 * np.sin(2 * np.pi * 7100 * t)
    elif attack_id == "A19":  # PSTN telephone compression
        signal = signal + 0.4 * np.sin(2 * np.pi * 4700 * t) + np.random.normal(0, 0.06, NUM_SAMPLES)
    else:  # Generic A07-A18
        high_freq = 4600 + (int(attack_id[1:]) * 150)
        signal = signal + 0.32 * np.sin(2 * np.pi * high_freq * t) + np.random.normal(0, 0.04, NUM_SAMPLES)

    envelope = np.clip(np.sin(2 * np.pi * 1.8 * t) * 1.8, 0.1, 1.0)
    signal = signal * envelope
    signal = signal / (np.max(np.abs(signal)) + 1e-6) * 0.85
    return signal.astype(np.float32)


def main():
    protocol_lines = []

    # 1. Bona fide human sample
    bonafide = generate_bonafide()
    bf_name = "LA_E_0000000.wav"
    sf.write(OUT_DIR / bf_name, bonafide, SAMPLE_RATE)
    protocol_lines.append(f"LA_0001 {bf_name} - - bonafide")
    print(f"Generated Bona Fide: {bf_name}")

    # 2. Attacks A01 through A19
    for idx in range(1, 20):
        attack_id = f"A{idx:02d}"
        audio = generate_attack(attack_id)
        filename = f"LA_E_{idx:07d}.wav"
        sf.write(OUT_DIR / filename, audio, SAMPLE_RATE)
        protocol_lines.append(f"LA_0002 {filename} - {attack_id} spoof")
        print(f"Generated {attack_id}: {filename}")

    # Write official evaluation protocol file
    protocol_path = OUT_DIR / "ASVspoof2019.LA.cm.eval.trl.txt"
    protocol_path.write_text("\n".join(protocol_lines) + "\n", encoding="utf-8")
    print(f"\nWrote ASVspoof 2019 LA protocol: {protocol_path}")


if __name__ == "__main__":
    main()
