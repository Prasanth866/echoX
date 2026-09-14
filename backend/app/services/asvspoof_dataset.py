"""
echoX - ASVspoof 2019 LA Dataset Service & Protocol Parser
Handles:
1. ASVspoof 2019 Logical Access (LA) protocol parsing (Train, Dev, Eval).
2. Attack algorithm definitions and metadata for A01 through A19.
3. Batch dataset loading for benchmarking and EER computation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

ASVSPOOF_2019_ATTACKS = {
    "bonafide": {
        "type": "Human",
        "name": "Bona Fide Human",
        "description": "Authentic natural human speech"
    },
    "A01": {
        "type": "TTS",
        "name": "Neural Acoustic + WaveNet",
        "description": "Neural acoustic model with WaveNet vocoder"
    },
    "A02": {
        "type": "TTS",
        "name": "Acoustic Model + WORLD",
        "description": "Neural acoustic model with WORLD transfer-function vocoder"
    },
    "A03": {
        "type": "TTS",
        "name": "Linear Prediction Vocoder",
        "description": "Acoustic model with classical LPC spectral envelope vocoder"
    },
    "A04": {
        "type": "TTS",
        "name": "Waveform Concatenation",
        "description": "Unit-selection concatenative speech synthesis"
    },
    "A05": {
        "type": "VC",
        "name": "Voice Conversion (VAE/GMM)",
        "description": "Variational autoencoder voice conversion with WORLD vocoder"
    },
    "A06": {
        "type": "VC",
        "name": "Spectral Filtering VC",
        "description": "Transfer-function voice conversion with spectral modification"
    },
    "A07": {
        "type": "TTS",
        "name": "Neural Vocoder (Unseen)",
        "description": "Unseen evaluation neural vocoder with high-frequency phase artifacts"
    },
    "A08": {
        "type": "TTS",
        "name": "Low Bit-rate Neural Codec",
        "description": "Neural network speech synthesis with sub-band quantization"
    },
    "A09": {
        "type": "TTS",
        "name": "Direct Waveform Generation",
        "description": "End-to-end direct waveform generative synthesis"
    },
    "A10": {
        "type": "TTS",
        "name": "Modern HiFi-GAN Vocoder",
        "description": "Tacotron 2 paired with multi-receptive field HiFi-GAN vocoder"
    },
    "A11": {
        "type": "TTS",
        "name": "Sinc-Filtered Vocoder",
        "description": "Bandpass filtered vocoder speech synthesis"
    },
    "A12": {
        "type": "TTS",
        "name": "LPC Residual Excitation",
        "description": "Linear predictive coding with glottal residual excitation"
    },
    "A13": {
        "type": "VC",
        "name": "Harmonic + Noise Model",
        "description": "Harmonic plus noise model (HNM) voice conversion"
    },
    "A14": {
        "type": "VC",
        "name": "Frequency Warping VC",
        "description": "Bilinear frequency warping voice conversion"
    },
    "A15": {
        "type": "VC",
        "name": "CycleGAN / StarGAN VC",
        "description": "Adversarial voice conversion with neural vocoding"
    },
    "A16": {
        "type": "TTS",
        "name": "High-Pitch Synthetic Speech",
        "description": "Formant-shifted high fundamental frequency synthesis"
    },
    "A17": {
        "type": "TTS",
        "name": "Low-Pitch Synthetic Speech",
        "description": "Low fundamental frequency synthesis with heavy sub-harmonics"
    },
    "A18": {
        "type": "TTS",
        "name": "Diffusion-Based Vocoder",
        "description": "Score-based diffusion acoustic synthesis"
    },
    "A19": {
        "type": "VC",
        "name": "PSTN/GSM Compressed Voice",
        "description": "Voice conversion passed through telephone bandwidth compression"
    }
}


class ProtocolEntry:
    def __init__(self, speaker_id: str, file_name: str, attack_id: str, key: str):
        self.speaker_id = speaker_id
        self.file_name = file_name
        self.attack_id = attack_id
        self.key = key  # "bonafide" or "spoof"
        self.is_spoof = (key.lower() == "spoof")


class ASVspoof2019LALoader:
    """
    Parser and batch loader for ASVspoof 2019 LA protocols.
    Standard protocol format:
    SPEAKER_ID AUDIO_FILENAME - SYSTEM_ID KEY
    e.g. LA_0023 LA_D_1000265 - - bonafide
    e.g. LA_0023 LA_D_1000266 - A01 spoof
    """

    @staticmethod
    def parse_protocol_file(protocol_path: Path) -> List[ProtocolEntry]:
        entries = []
        with open(protocol_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    speaker_id = parts[0]
                    file_name = parts[1]
                    attack_id = parts[3]
                    key = parts[4]
                    entries.append(ProtocolEntry(speaker_id, file_name, attack_id, key))
        return entries

    @staticmethod
    def get_attack_info(attack_id: str) -> Dict[str, str]:
        return ASVSPOOF_2019_ATTACKS.get(
            attack_id,
            {"type": "Unknown", "name": f"Attack {attack_id}", "description": "Unclassified attack algorithm"}
        )
