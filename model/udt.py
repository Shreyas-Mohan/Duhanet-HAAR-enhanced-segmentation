import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List

class UncertaintyAwareDualTraining:
    """
    Uncertainty-aware Dual-branch Training (UDT) Strategy

    Key features:
        - Uncertainty estimation from dual-branch disagreement and entropy
        - Frequency-domain uncertainty enhancement using Haar features
        - Adaptive sample weighting based on uncertainty
        - Dual-branch consistency regularization
        - Progressive thresholding for reliable samples
    """

    def __init__(
        self,
        num_classes: int,
        consistency_weight: float = 0.4,
        entropy_weight: float = 0.4,
        haar_weight: float = 0.2,
        uncertainty_loss_weight: float = 0.2,
        consistency_loss_weight: float = 0.1,
        base_threshold: float = 0.7,
        warmup_epochs: int = 5,
        weight_type: str = "exp",
        verbose: bool = False
    ):
        self.num_classes = num_classes
        self.verbose = verbose

        # Uncertainty estimation weights
        self.consistency_weight = consistency_weight
        self.entropy_weight = entropy_weight
        self.haar_weight = haar_weight

        # Loss weights
        self.uncertainty_loss_weight = uncertainty_loss_weight
        self.consistency_loss_weight = consistency_loss_weight

        # Progressive thresholding
        self.base_threshold = base_threshold
        self.warmup_epochs = warmup_epochs
        self.weight_type = weight_type

        # Training state & Statistics
        self.current_epoch = 0
        self.uncertainty_history: List[float] = []
        self.consistency_history: List[float] = []

        if verbose:
            print(f"✅ UDT initialized (weight_type={weight_type})")

    # ==================== Uncertainty Estimation ====================

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to [0, 1] range."""
        x_min, x_max = x.min(), x.max()
        if x_max > x_min:
            return (x - x_min) / (x_max - x_min + 1e-8)
        return x

    def _compute_haar_uncertainty(self, haar_features: Optional[List[torch.Tensor]], target_size: torch.Size) -> Optional[torch.Tensor]:
        """Compute frequency-domain uncertainty from Haar features."""
        if not haar_features:
            return None

        # Use the deepest Haar feature for frequency analysis
        feat = haar_features[-1]

        # Align spatial dimensions
        if feat.shape[2:] != target_size:
            feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)

        # Compute gradients (high-frequency components) using slicing for memory efficiency
        grad_x = torch.abs(feat[..., 1:] - feat[..., :-1]).mean(dim=1)
        grad_y = torch.abs(feat[:, :, 1:, :] - feat[:, :, :-1, :]).mean(dim=1)

        # Pad to maintain spatial dimensions
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))

        return (grad_x + grad_y) * 0.5

    def estimate_uncertainty(
        self,
        local_probs: torch.Tensor,
        global_probs: torch.Tensor,
        haar_features: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate pixel-wise uncertainty from dual-branch predictions."""
        # 1. Branch disagreement
        disagreement = torch.abs(local_probs - global_probs).mean(dim=1)

        # 2. Prediction entropy (using fused predictions)
        fused_probs = (local_probs + global_probs) * 0.5
        entropy = -torch.sum(fused_probs * torch.log(fused_probs.clamp(min=1e-10)), dim=1)
        
        # Dynamically allocate the max_entropy tensor to the same device as the inputs
        max_entropy = torch.log(torch.tensor(self.num_classes, dtype=torch.float32, device=local_probs.device))
        normalized_entropy = entropy / max_entropy

        # Combine disagreement and entropy
        uncertainty = (self.consistency_weight * disagreement) + (self.entropy_weight * normalized_entropy)

        # 3. Frequency-domain uncertainty (Haar features)
        if haar_features is not None:
            haar_uncertainty = self._compute_haar_uncertainty(haar_features, uncertainty.shape[1:])
            if haar_uncertainty is not None:
                uncertainty += self.haar_weight * haar_uncertainty

        # Normalize to [0, 1]
        uncertainty = self._normalize(uncertainty).clamp(0, 1)
        confidence = 1.0 - uncertainty

        return uncertainty, confidence

    # ==================== Uncertainty Weighting ====================

    def _get_weight(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """Convert uncertainty to sample weight."""
        if self.weight_type == "exp":
            return torch.exp(-uncertainty)
        elif self.weight_type == "square":
            return (1.0 - uncertainty) ** 2
        return 1.0 - uncertainty  # linear fallback

    # ==================== Loss Computation ====================

    def compute_consistency_loss(self, local_logits: torch.Tensor, global_logits: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        """Compute dual-branch consistency loss with uncertainty weighting."""
        if local_logits.shape[2:] != global_logits.shape[2:]:
            global_logits = F.interpolate(global_logits, size=local_logits.shape[2:], mode='bilinear', align_corners=False)

        local_probs = F.softmax(local_logits, dim=1)
        global_probs = F.softmax(global_logits, dim=1)

        # Symmetric KL divergence
        kl_local_to_global = F.kl_div(local_probs.log(), global_probs, reduction='none').sum(1)
        kl_global_to_local = F.kl_div(global_probs.log(), local_probs, reduction='none').sum(1)
        divergence = (kl_local_to_global + kl_global_to_local) * 0.5

        weight = self._get_weight(uncertainty)
        loss = (divergence * weight).mean()

        return self.consistency_loss_weight * loss

    def compute_uncertainty_loss(self, logits: torch.Tensor, target: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss with uncertainty-aware weighting."""
        ce_loss = F.cross_entropy(logits, target, reduction='none')
        weight = self._get_weight(uncertainty)

        # Progressive thresholding (after warmup)
        if self.current_epoch >= self.warmup_epochs:
            reliable_mask = ((1.0 - uncertainty) > self.base_threshold).float()
            weight = weight * reliable_mask

        weighted_loss = (ce_loss * weight).sum() / (weight.sum() + 1e-6)
        return self.uncertainty_loss_weight * weighted_loss

    # ==================== Main Forward ====================

    def forward(self, model: torch.nn.Module, images: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Perform one forward pass with UDT supervision."""
        predictions, features = model.forward_with_features(images)
        main_output = predictions[0] if isinstance(predictions, (tuple, list)) else predictions
        
        total_loss = F.cross_entropy(main_output, targets)

        if features is None or not hasattr(features, 'local_logits'):
            return total_loss, None, None, main_output

        local_logits = features.local_logits
        global_logits = features.global_logits

        local_probs = F.softmax(local_logits, dim=1)
        global_probs = F.softmax(global_logits, dim=1)

        uncertainty, confidence = self.estimate_uncertainty(
            local_probs, global_probs, getattr(features, 'haar_features', None)
        )

        uncertainty_weighted_loss = self.compute_uncertainty_loss(main_output, targets, uncertainty)
        total_loss += uncertainty_weighted_loss

        consistency_loss = self.compute_consistency_loss(local_logits, global_logits, uncertainty)
        total_loss += consistency_loss

        # Track statistics
        self.uncertainty_history.append(uncertainty.mean().item())
        self.consistency_history.append(consistency_loss.item())

        return total_loss, uncertainty, confidence, main_output

    # ==================== Training Control ====================

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def get_stats(self) -> Dict[str, float]:
        return {
            "epoch": self.current_epoch,
            "uncertainty": np.mean(self.uncertainty_history[-100:]) if self.uncertainty_history else 0.0,
            "consistency_loss": np.mean(self.consistency_history[-100:]) if self.consistency_history else 0.0
        }

    def reset_stats(self):
        self.uncertainty_history.clear()
        self.consistency_history.clear()


# ==================== Training Helper ====================

class UDTTrainer:
    """Training wrapper for DUHA-Net with UDT strategy."""

    def __init__(self, model: torch.nn.Module, udt_strategy: UncertaintyAwareDualTraining, optimizer: torch.optim.Optimizer):
        self.model = model
        self.udt = udt_strategy
        self.optimizer = optimizer
        # Dynamically fetch the device from the model parameters
        self.device = next(model.parameters()).device 

    def set_epoch(self, epoch: int):
        self.udt.set_epoch(epoch)

    def train_step(self, images: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        loss, uncertainty, _, _ = self.udt.forward(self.model, images, targets)

        loss.backward()
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'uncertainty': uncertainty.mean().item() if uncertainty is not None else 0.0,
            'stats': self.udt.get_stats()
        }

    @torch.no_grad()
    def eval_step(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        images = images.to(self.device, non_blocking=True)

        predictions = self.model(images)
        if isinstance(predictions, dict):
            predictions = predictions.get('segmentation', predictions.get('seg'))

        return predictions