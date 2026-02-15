"""
Task-guided contrastive learning modules for RDFNet
Implements:
- SeverityHead: predicts fog severity for adaptive λ(x)
- SpatialWeightHead: produces per-pixel weight maps for feature alignment
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeverityHead(nn.Module):
    """
    Predicts fog severity from backbone features.
    Input: feature map (B, C, H, W)
    Output: severity scalar s(x) per image (B, 1)
    """
    def __init__(self, in_channels, hidden_dim=128):
        super(SeverityHead, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            s: (B, 1) severity scores
        """
        # Global average pooling
        x = self.gap(x)  # (B, C, 1, 1)
        x = x.view(x.size(0), -1)  # (B, C)
        s = self.mlp(x)  # (B, 1)
        return s


class SpatialWeightHead(nn.Module):
    """
    Produces per-pixel weight maps for spatially adaptive feature alignment.
    Uses 1x1 convolutions to generate weights from features themselves.
    """
    def __init__(self, in_channels, mid_channels=None):
        super(SpatialWeightHead, self).__init__()
        if mid_channels is None:
            mid_channels = max(in_channels // 4, 16)
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            w: (B, 1, H, W) spatial weight map in [0, 1]
        """
        w = self.conv1(x)
        w = self.bn(w)
        w = self.act(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        return w


def compute_adaptive_lambda(severity, lambda_min=0.05, lambda_max=0.20):
    """
    Compute adaptive loss weight λ(x) from severity score s(x).
    
    Formula: λ(x) = λ_min + (λ_max - λ_min) * σ(s(x))
    
    Args:
        severity: (B, 1) severity scores
        lambda_min: minimum λ value
        lambda_max: maximum λ value
    
    Returns:
        lambda_adaptive: (B, 1) adaptive weights in [λ_min, λ_max]
    """
    # Sigmoid to bound between 0 and 1
    sigma_s = torch.sigmoid(severity)
    
    # Linear mapping to [λ_min, λ_max]
    lambda_adaptive = lambda_min + (lambda_max - lambda_min) * sigma_s
    
    return lambda_adaptive


class MultiScaleSpatialWeights(nn.Module):
    """
    Wrapper to generate spatial weights for multiple feature scales (P3, P4, P5).
    """
    def __init__(self, channels_list):
        """
        Args:
            channels_list: list of channels for each scale, e.g., [128, 256, 512]
        """
        super(MultiScaleSpatialWeights, self).__init__()
        self.weight_heads = nn.ModuleList([
            SpatialWeightHead(c) for c in channels_list
        ])
    
    def forward(self, features):
        """
        Args:
            features: list of feature maps [(B,C1,H1,W1), (B,C2,H2,W2), (B,C3,H3,W3)]
        Returns:
            weights: list of weight maps [(B,1,H1,W1), (B,1,H2,W2), (B,1,H3,W3)]
        """
        weights = [head(feat) for head, feat in zip(self.weight_heads, features)]
        return weights
