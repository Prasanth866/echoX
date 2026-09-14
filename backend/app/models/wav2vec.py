"""
EchoX - Secondary Acoustic Feature Backbone (Wav2Vec / SSL Representation)
Provides auxiliary acoustic and self-supervised feature extraction for multi-tier fusion.
"""

import torch
import torch.nn as nn


class Wav2VecBackbone(nn.Module):
    """
    Lightweight acoustic feature extractor mimicking the convolutional feature encoder
    of Wav2Vec 2.0. Used as a secondary representation backbone alongside AASIST.
    """

    def __init__(self, out_features: int = 64):
        super(Wav2VecBackbone, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=10, stride=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, out_features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(out_features),
            nn.GELU(),
        )
        self.proj = nn.Linear(out_features, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (batch, samples) or (batch, 1, samples)
        Output: (batch, 32) feature embedding
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feats = self.encoder(x)
        pooled = feats.mean(dim=-1)
        return self.proj(pooled)
