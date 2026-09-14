"""
VoiceGuard AI - AASIST Architecture
Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks
Based on Jung et al. (Interspeech 2021 / Clova AI Research)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv(nn.Module):
    """
    Sinc-based convolution layer directly operating on raw waveform audio.
    Applies learnable bandpass sinc filters to extract spectro-temporal features.
    """

    def __init__(
        self,
        out_channels: int = 70,
        kernel_size: int = 128,
        sample_rate: int = 16000,
        in_channels: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        groups: int = 1,
        min_low_hz: int = 50,
        min_band_hz: int = 50
    ):
        super(SincConv, self).__init__()

        if in_channels != 1:
            err = f"SincConv only supports in_channels=1, got {in_channels}"
            raise ValueError(err)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = torch.linspace(
            self._to_mel(low_hz),
            self._to_mel(high_hz),
            self.out_channels + 1
        )
        hz = self._to_hz(mel)

        self.low_hz_ = nn.Parameter(hz[:-1].view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.diff(hz).view(-1, 1))

        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate

    @staticmethod
    def _to_mel(hz):
        return 2595 * math.log10(1 + hz / 700)

    @staticmethod
    def _to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Waveforms: (batch, 1, samples)
        Returns: (batch, out_channels, time_steps)
        """
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])

        filters = band_pass.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(
            waveforms,
            filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1
        )


class ResidualBlock(nn.Module):
    """Residual Block with Batch Normalization and Max Pooling"""

    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        out = F.leaky_relu(out + res, 0.2)
        return self.pool(out)


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer for Spectro-Temporal interactions"""

    def __init__(self, in_features: int = 64, out_features: int = 32):
        super(GraphAttentionLayer, self).__init__()
        self.q = nn.Linear(in_features, out_features)
        self.k = nn.Linear(in_features, out_features)
        self.v = nn.Linear(in_features, out_features)
        self.scale = 1.0 / math.sqrt(out_features)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        Q = self.q(h)
        K = self.k(h)
        V = self.v(h)
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.bmm(attn, V)
        return F.elu(out)


class AASIST(nn.Module):
    """
    AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks.
    Outputs: [logits_bonafide, logits_spoof] (dim: batch_size x 2)
    """

    def __init__(self, num_classes: int = 2):
        super(AASIST, self).__init__()
        self.sinc_conv = SincConv(out_channels=70, kernel_size=128, in_channels=1)

        self.res1 = ResidualBlock(1, 32)
        self.res2 = ResidualBlock(32, 64)

        self.gat_layer = GraphAttentionLayer(in_features=64, out_features=32)

        self.fc_head = nn.Sequential(
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (batch, samples) or (batch, 1, samples) where samples = 64,600
        Output: (batch, 2) logits: [bonafide, spoof]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        sinc_feats = torch.abs(self.sinc_conv(x))

        feats = sinc_feats.unsqueeze(1)

        c_feats = self.res1(feats)
        c_feats = self.res2(c_feats)

        freq_pooled = c_feats.mean(dim=2)
        graph_nodes = F.interpolate(freq_pooled, size=64, mode="linear", align_corners=False).permute(0, 2, 1)

        gat_out = self.gat_layer(graph_nodes)

        pooled = gat_out.mean(dim=1)

        logits = self.fc_head(pooled)
        return logits

