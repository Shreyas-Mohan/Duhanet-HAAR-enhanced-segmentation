import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class EfficientAttentionGate(nn.Module):
    """
    Efficient Attention Gate (EAG) for adaptive feature fusion.
    Uses grouped convolutions to reduce parameter count and computational overhead.
    """
    def __init__(self, F_g: int, F_l: int, F_int: int, num_groups: int = 8):
        super().__init__()
        
        # Ensure groups do not exceed channel dimensions
        g_groups = min(num_groups, max(1, F_g))
        x_groups = min(num_groups, max(1, F_l))

        self.grouped_conv_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, groups=g_groups),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True)
        )
        
        self.grouped_conv_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, groups=x_groups),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Align spatial dimensions dynamically
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
            
        g1 = self.grouped_conv_g(g)
        x1 = self.grouped_conv_x(x)
        
        # In-place ReLU for memory efficiency
        fusion = F.relu(x1 + g1, inplace=True)
        psi = self.psi(fusion)
        
        return (x * psi) + x


class EfficientChannelAttention(nn.Module):
    """
    Efficient Channel Attention (ECA) module.
    Dynamically determines kernel size based on channel dimensions.
    """
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        # Calculate adaptive kernel size
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze spatial dimensions and transpose for Conv1d
        v = self.avg_pool(x).squeeze(-1).transpose(1, 2)
        v = self.conv(v)
        # Restore dimensions
        v = v.transpose(1, 2).unsqueeze(-1)
        return self.sigmoid(v)


class SemanticGuidedDecouplingAggregation(nn.Module):
    """
    Semantic-Guided Decoupling Aggregation (S-GDA)
    Explicitly separates overlapping semantic features between local and global branches.
    """
    def __init__(self, in_channels: int, num_classes: int, mid_channels: int = None, num_groups: int = 8):
        super().__init__()
        mid = mid_channels if mid_channels is not None else max(in_channels // 2, 1)

        # 1. Efficient Attention Gate
        self.eag = EfficientAttentionGate(in_channels, in_channels, mid, num_groups)

        # 2. Inter-Class Decoupling Components
        self.class_attn_generator = nn.Sequential(
            nn.Conv2d(in_channels * 2, num_classes, kernel_size=1),
            nn.Softmax(dim=1)
        )
        self.background_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.class_aggregation = nn.Sequential(
            nn.Conv2d(in_channels * num_classes, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 3. Efficient Channel Attention
        self.eca = EfficientChannelAttention(in_channels * 2)

        # 4. Final projection
        self.project = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, global_feat: torch.Tensor, local_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Step 1: Adaptive fusion via EAG
        gated_local = self.eag(global_feat, local_feat)

        if global_feat.shape[2:] != gated_local.shape[2:]:
            global_feat = F.interpolate(global_feat, size=gated_local.shape[2:], mode='bilinear', align_corners=False)

        # Step 2: Inter-Class Decoupling
        concat = torch.cat([gated_local, global_feat], dim=1)
        class_attn = self.class_attn_generator(concat)
        bg_gate = self.background_gate(concat)

        # Fast Vectorized Decoupling (Replaces the slow Python for-loop)
        # adjusted_attn: [B, num_classes, H, W]
        adjusted_attn = class_attn * (1.0 - bg_gate)
        
        B, C, H, W = gated_local.shape
        num_classes = adjusted_attn.shape[1]
        
        # Broadcasting: [B, 1, C, H, W] * [B, num_classes, 1, H, W] -> [B, num_classes, C, H, W]
        decoupled = gated_local.view(B, 1, C, H, W) * adjusted_attn.view(B, num_classes, 1, H, W)
        
        # Flatten class and channel dimensions -> [B, num_classes * C, H, W]
        decoupled = decoupled.reshape(B, num_classes * C, H, W)

        # Aggregate decoupled features
        decoupled_out = self.class_aggregation(decoupled)
        enhanced_local = gated_local + (self.alpha * decoupled_out)

        # Step 3: Channel attention
        final_concat = torch.cat([enhanced_local, global_feat], dim=1)
        ca_weight = self.eca(final_concat)
        final_concat = final_concat * ca_weight

        # Step 4: Final projection
        return self.project(final_concat), class_attn