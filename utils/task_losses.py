"""
Task-guided loss functions for feature-centric supervision.
Implements:
- Feature Alignment Loss (weighted MSE)
- InfoNCE Contrastive Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAlignmentLoss(nn.Module):
    """
    Spatially-weighted feature alignment loss.
    Computes MSE between features weighted by spatial importance maps.
    """
    def __init__(self):
        super(FeatureAlignmentLoss, self).__init__()
    
    def forward(self, pred_features, target_features, spatial_weights):
        """
        Args:
            pred_features: list of [(B, C, H, W)] predicted features at multiple scales
            target_features: list of [(B, C, H, W)] target features (detached)
            spatial_weights: list of [(B, 1, H, W)] weight maps
        
        Returns:
            loss: scalar alignment loss
        """
        total_loss = 0.0
        num_scales = len(pred_features)
        
        for pred, target, weight in zip(pred_features, target_features, spatial_weights):
            # Ensure target is detached (stop gradient)
            target = target.detach()
            
            # Normalize weights to prevent collapse
            weight = weight / (weight.mean() + 1e-6)
            
            # Compute weighted MSE
            diff = (pred - target) ** 2  # (B, C, H, W)
            weighted_diff = diff * weight  # (B, C, H, W) * (B, 1, H, W)
            
            # Average over all dimensions
            scale_loss = weighted_diff.mean()
            total_loss += scale_loss
        
        # Average across scales
        return total_loss / num_scales


class InfoNCEContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for multi-scale feature invariance.
    Pulls together features from two views of the same image,
    pushes apart features from different images.
    """
    def __init__(self, temperature=0.2):
        super(InfoNCEContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, features_v1, features_v2):
        """
        Args:
            features_v1: list of [(B, C, H, W)] features from view 1
            features_v2: list of [(B, C, H, W)] features from view 2
        
        Returns:
            loss: scalar InfoNCE loss
        """
        # Pool features from all scales
        z_v1 = self._pool_multiscale_features(features_v1)  # (B, D)
        z_v2 = self._pool_multiscale_features(features_v2)  # (B, D)
        
        # L2 normalize
        z_v1 = F.normalize(z_v1, dim=1)
        z_v2 = F.normalize(z_v2, dim=1)
        
        batch_size = z_v1.size(0)
        
        # Compute similarity matrix (B, B)
        # sim[i, j] = cosine_similarity(z_v1[i], z_v2[j])
        similarity_matrix = torch.matmul(z_v1, z_v2.T) / self.temperature
        
        # Positive pairs are on the diagonal (same image, different views)
        # Create labels: [0, 1, 2, ..., B-1]
        labels = torch.arange(batch_size, device=z_v1.device)
        
        # Cross entropy loss (InfoNCE)
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def _pool_multiscale_features(self, features):
        """
        Global average pool features from multiple scales and concatenate.
        
        Args:
            features: list of [(B, C_i, H_i, W_i)]
        
        Returns:
            pooled: (B, sum(C_i)) concatenated pooled features
        """
        pooled = []
        for feat in features:
            # Global average pooling
            gap = F.adaptive_avg_pool2d(feat, 1)  # (B, C, 1, 1)
            gap = gap.view(gap.size(0), -1)  # (B, C)
            pooled.append(gap)
        
        # Concatenate across scales
        return torch.cat(pooled, dim=1)  # (B, C1+C2+C3)


def compute_task_guided_loss(
    detection_loss,
    pred_features_v1,
    target_features,
    spatial_weights,
    features_v1,
    features_v2,
    severity_scores,
    lambda_min=0.05,
    lambda_max=0.20,
    beta=0.1,
    align_criterion=None,
    contrast_criterion=None
):
    """
    Compute total task-guided loss.
    
    L_total = L_det + λ(x) * L_align + β * L_con
    
    Args:
        detection_loss: scalar, YOLOv7 detection loss
        pred_features_v1: list of predicted features from view 1
        target_features: list of target features (clean or view2)
        spatial_weights: list of spatial weight maps
        features_v1: list of features from view 1 for contrastive
        features_v2: list of features from view 2 for contrastive
        severity_scores: (B, 1) severity predictions
        lambda_min: minimum λ
        lambda_max: maximum λ
        beta: weight for contrastive loss
        align_criterion: FeatureAlignmentLoss instance
        contrast_criterion: InfoNCEContrastiveLoss instance
    
    Returns:
        total_loss: scalar
        loss_dict: dictionary with individual losses for logging
    """
    # Initialize criteria if not provided
    if align_criterion is None:
        align_criterion = FeatureAlignmentLoss()
    if contrast_criterion is None:
        contrast_criterion = InfoNCEContrastiveLoss()
    
    # Compute adaptive λ(x)
    from nets.task_modules import compute_adaptive_lambda
    lambda_adaptive = compute_adaptive_lambda(severity_scores, lambda_min, lambda_max)
    
    # Compute alignment loss
    L_align = align_criterion(pred_features_v1, target_features, spatial_weights)
    
    # Compute contrastive loss
    L_con = contrast_criterion(features_v1, features_v2)
    
    # Apply adaptive weighting (average across batch)
    lambda_mean = lambda_adaptive.mean()
    
    # Total loss
    total_loss = detection_loss + lambda_mean * L_align + beta * L_con
    
    # Loss dictionary for logging
    loss_dict = {
        'total': total_loss.item(),
        'detection': detection_loss.item(),
        'alignment': L_align.item(),
        'contrastive': L_con.item(),
        'lambda_mean': lambda_mean.item(),
    }
    
    return total_loss, loss_dict
