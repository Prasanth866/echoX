"""
echoX - Facebook Wav2Vec 2.0 Deepfake & Anti-Spoofing Neural Backbone
Utilizes Facebook / Meta's Wav2Vec 2.0 architecture for self-supervised speech
representations and deepfake voice detection.
"""

import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Wav2Vec2Model

from backend.app.core.config import MODEL_CONFIG, AUDIO_CONFIG


class FacebookWav2Vec2Detector(nn.Module):
    """
    Facebook Wav2Vec 2.0 Neural Backbone for Voice Deepfake and Anti-Spoofing Detection.
    Extracts high-resolution latent representations and evaluates synthetic speech probability.
    """

    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or self._select_device()
        self.model_name = MODEL_CONFIG.FINE_TUNED_MODEL
        self.primary_name = MODEL_CONFIG.PRIMARY_MODEL
        self.feature_extractor = None
        self.classifier_model = None
        self.base_model = None
        self._initialize_model()

    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _initialize_model(self):
        """
        Loads Facebook Wav2Vec 2.0 weights.
        First attempts to load the fine-tuned Wav2Vec 2.0 sequence classifier.
        Falls back to facebook/wav2vec2-base if needed.
        """
        print(f"[Wav2Vec2] Initializing Facebook Wav2Vec 2.0 on device: {self.device}")
        
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_name,
                sampling_rate=AUDIO_CONFIG.SAMPLE_RATE
            )
            self.classifier_model = AutoModelForAudioClassification.from_pretrained(
                self.model_name
            ).to(self.device)
            self.classifier_model.eval()
            print(f"[Wav2Vec2] Loaded fine-tuned Facebook Wav2Vec 2.0 classifier: {self.model_name}")
            return
        except Exception as e:
            print(f"[Wav2Vec2] Could not load fine-tuned checkpoint ({e}). Trying base {self.primary_name}...")

        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.primary_name,
                sampling_rate=AUDIO_CONFIG.SAMPLE_RATE
            )
            self.base_model = Wav2Vec2Model.from_pretrained(self.primary_name).to(self.device)
            self.base_model.eval()
            
            self.head = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 2)
            ).to(self.device)
            self.head.eval()
            print(f"[Wav2Vec2] Initialized base Facebook Wav2Vec 2.0: {self.primary_name}")
        except Exception as e:
            print(f"[Wav2Vec2] Critical error loading Wav2Vec2: {e}")
            raise RuntimeError(f"Failed to load Facebook Wav2Vec 2.0: {e}")

    def forward(self, audio_array: np.ndarray, sr: int = 16000) -> Tuple[float, float, Dict[str, Any]]:
        """
        Performs forward-pass inference on audio waveform.
        Returns:
            spoof_prob (float): Probability of synthetic / cloned voice [0.0, 1.0]
            bonafide_prob (float): Probability of authentic human voice [0.0, 1.0]
            latents_meta (dict): Latent feature statistics
        """
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)
        audio_array = audio_array.astype(np.float32)

        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=sr,
            return_tensors="pt"
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            if self.classifier_model is not None:
                outputs = self.classifier_model(input_values, output_hidden_states=True)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
                
                id2label = getattr(self.classifier_model.config, "id2label", {0: "fake", 1: "real"})
                if id2label.get(0, "").lower() in ["fake", "spoof"]:
                    spoof_prob = float(probs[0])
                    bonafide_prob = float(probs[1])
                else:
                    bonafide_prob = float(probs[0])
                    spoof_prob = float(probs[1])

                if outputs.hidden_states:
                    last_hidden = outputs.hidden_states[-1]
                    feat_variance = float(torch.var(last_hidden).cpu())
                    feat_norm = float(torch.norm(last_hidden, dim=-1).mean().cpu())
                else:
                    feat_variance = 0.0
                    feat_norm = 0.0
            else:
                outputs = self.base_model(input_values)
                hidden_states = outputs.last_hidden_state
                pooled = hidden_states.mean(dim=1)
                logits = self.head(pooled)
                probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
                spoof_prob = float(probs[0])
                bonafide_prob = float(probs[1])
                feat_variance = float(torch.var(hidden_states).cpu())
                feat_norm = float(torch.norm(pooled).cpu())

        latents_meta = {
            "model_architecture": "Facebook Wav2Vec 2.0",
            "hidden_dimension": 768,
            "latent_norm": round(feat_norm, 3),
            "latent_variance": round(feat_variance, 4)
        }

        return spoof_prob, bonafide_prob, latents_meta
