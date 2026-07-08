"""Remote-sensing multi-scale attention adapter RoI head.

This v2 adapter strengthens RoI feature extraction with remote-sensing
structure cues before RVLP-LCC classification. It combines lightweight ASPP,
coordinate attention, ECA, and optional SimAM without changing the detector
interface.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS

from .rsf_adapter_roi_head import RSFAdapterRoIHead


def _gn_groups(channels):
    if channels % 32 == 0:
        return 32
    if channels % 16 == 0:
        return 16
    if channels % 8 == 0:
        return 8
    return 1


class ECALayer(nn.Module):
    """Efficient channel attention with very few parameters."""

    def __init__(self, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class CoordAttention(nn.Module):
    """Coordinate attention for direction-aware remote-sensing RoI features."""

    def __init__(self, channels=256, reduction=32):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_gn_groups(hidden), hidden)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(hidden, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(hidden, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.norm(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)
        a_h = self.sigmoid(self.conv_h(y_h))
        a_w = self.sigmoid(self.conv_w(y_w))
        return x * a_h * a_w


class SimAMLayer(nn.Module):
    """Parameter-free spatial attention for low-shot regularization."""

    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = float(e_lambda)

    def forward(self, x):
        n = x.size(2) * x.size(3) - 1
        if n <= 0:
            return x
        mean = x.mean(dim=(2, 3), keepdim=True)
        energy = (x - mean).pow(2)
        denom = 4 * (energy.sum(dim=(2, 3), keepdim=True) / n +
                     self.e_lambda)
        return x * torch.sigmoid(energy / denom + 0.5)


class ASPPLite(nn.Module):
    """Lightweight ASPP using depthwise separable branches."""

    def __init__(self,
                 channels=256,
                 hidden_channels=128,
                 dilations=(1, 2, 3),
                 dropout=0.0):
        super().__init__()
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=channels,
                        bias=False),
                    nn.Conv2d(
                        channels, hidden_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(_gn_groups(hidden_channels),
                                 hidden_channels),
                    nn.ReLU(inplace=True)))
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))
        fused_channels = hidden_channels * (len(dilations) + 1)
        self.project = nn.Sequential(
            nn.Conv2d(fused_channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_gn_groups(channels), channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(float(dropout)))

    def forward(self, x):
        size = x.shape[-2:]
        feats = [branch(x) for branch in self.branches]
        pooled = F.interpolate(
            self.image_pool(x), size=size, mode='bilinear', align_corners=False)
        feats.append(pooled)
        return self.project(torch.cat(feats, dim=1))


class RSFeatureAdapterV2(nn.Module):
    """Coord-ECA-ASPP adapter for few-shot remote-sensing detection."""

    def __init__(self,
                 channels=256,
                 hidden_channels=128,
                 aspp_dilations=(1, 2, 3),
                 dropout=0.0,
                 residual_scale_init=0.2,
                 eca_kernel_size=3,
                 coord_reduction=32,
                 use_coord=True,
                 use_eca=True,
                 use_simam=True):
        super().__init__()
        self.aspp = ASPPLite(
            channels=channels,
            hidden_channels=hidden_channels,
            dilations=tuple(aspp_dilations),
            dropout=dropout)
        self.coord = CoordAttention(
            channels=channels, reduction=coord_reduction) if use_coord else None
        self.eca = ECALayer(kernel_size=eca_kernel_size) if use_eca else None
        self.simam = SimAMLayer() if use_simam else None
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init)))

    def forward(self, x):
        residual = self.aspp(x)
        if self.coord is not None:
            residual = self.coord(residual)
        if self.eca is not None:
            residual = self.eca(residual)
        if self.simam is not None:
            residual = self.simam(residual)
        return x + self.residual_scale.tanh() * residual


@HEADS.register_module()
class RSFAdapterV2RoIHead(RSFAdapterRoIHead):
    """RSF v2 RoI head using Coord-ECA-ASPP feature adaptation."""

    def __init__(self,
                 adapter_channels=256,
                 adapter_hidden_channels=128,
                 adapter_aspp_dilations=(1, 2, 3),
                 adapter_dropout=0.0,
                 adapter_residual_scale_init=0.2,
                 adapter_eca_kernel_size=3,
                 adapter_coord_reduction=32,
                 adapter_use_coord=True,
                 adapter_use_eca=True,
                 adapter_use_simam=True,
                 **kwargs):
        super().__init__(
            adapter_channels=adapter_channels,
            adapter_hidden_channels=adapter_hidden_channels,
            adapter_dropout=adapter_dropout,
            adapter_residual_scale_init=adapter_residual_scale_init,
            **kwargs)
        self.rsf_adapter = RSFeatureAdapterV2(
            channels=int(adapter_channels),
            hidden_channels=int(adapter_hidden_channels),
            aspp_dilations=adapter_aspp_dilations,
            dropout=float(adapter_dropout),
            residual_scale_init=float(adapter_residual_scale_init),
            eca_kernel_size=int(adapter_eca_kernel_size),
            coord_reduction=int(adapter_coord_reduction),
            use_coord=adapter_use_coord,
            use_eca=adapter_use_eca,
            use_simam=adapter_use_simam)
