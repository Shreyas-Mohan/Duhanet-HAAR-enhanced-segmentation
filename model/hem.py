import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class HaarEnhancementModule(nn.Module):
    """
    Haar Enhancement Module (HEM)
    Enhances boundary precision using Haar wavelet transforms in the frequency domain.
    
    Architecture flow:
        DWT (Haar) -> Channel Compression -> Non-linear Processing -> Channel Recovery -> IDWT
    """

    def __init__(self, channels: int, reduction_ratio: int = 4):
        super().__init__()

        # Ensure we don't compress down to 0 channels
        compressed_channels = max(channels // reduction_ratio, 4)

        # Unified compression and expansion for all 4 sub-bands (LL, LH, HL, HH)
        self.compressors = nn.ModuleList([
            self._make_compressor(channels, compressed_channels) for _ in range(4)
        ])
        
        self.expanders = nn.ModuleList([
            nn.Conv2d(compressed_channels, channels, kernel_size=1) for _ in range(4)
        ])

        # Shared non-linear processing block
        self.process_block = nn.Sequential(
            nn.Conv2d(compressed_channels, compressed_channels, kernel_size=3, padding=1, groups=compressed_channels),
            nn.BatchNorm2d(compressed_channels),
            nn.GELU(),
            nn.Conv2d(compressed_channels, compressed_channels, kernel_size=1),
            nn.BatchNorm2d(compressed_channels),
            nn.GELU()
        )

        # Register Haar kernels as non-trainable buffers (automatically moved to correct device)
        self.register_buffer('haar_kernel', self._get_haar_kernel(channels))
        self.register_buffer('inv_haar_kernel', self._get_inverse_haar_kernel(channels))

        # Learnable residual weight
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        # Output normalization
        self.out_norm = nn.BatchNorm2d(channels)
        self.out_act = nn.ReLU(inplace=True)

    @staticmethod
    def _make_compressor(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    @staticmethod
    def _get_haar_kernel(channels: int) -> torch.Tensor:
        """Generates the standard 2x2 Haar Wavelet Decomposition kernels."""
        LL = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        LH = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32)
        HL = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.float32)
        HH = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)

        base_kernels = torch.stack([LL, LH, HL, HH], dim=0).unsqueeze(1)
        return base_kernels.repeat(channels, 1, 1, 1)

    @staticmethod
    def _get_inverse_haar_kernel(channels: int) -> torch.Tensor:
        """Generates the standard 2x2 Haar Wavelet Reconstruction kernels."""
        inv_LL = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        inv_LH = torch.tensor([[0.5, 0.5], [-0.5, -0.5]], dtype=torch.float32)
        inv_HL = torch.tensor([[0.5, -0.5], [0.5, -0.5]], dtype=torch.float32)
        inv_HH = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)

        base_kernels = torch.stack([inv_LL, inv_LH, inv_HL, inv_HH], dim=0).unsqueeze(1)
        return base_kernels.repeat(channels, 1, 1, 1)

    def dwt(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs Discrete Wavelet Transform via strided convolution."""
        B, C, H, W = x.shape
        
        # Pad if dimensions are odd
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2), mode='reflect')

        x_dwt = F.conv2d(x, self.haar_kernel, stride=2, groups=C)
        x_dwt = x_dwt.reshape(B, C, 4, x_dwt.shape[2], x_dwt.shape[3])

        return x_dwt[:, :, 0], x_dwt[:, :, 1], x_dwt[:, :, 2], x_dwt[:, :, 3]

    def idwt(self, sub_bands: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Performs Inverse Discrete Wavelet Transform via transposed convolution."""
        x_stacked = torch.cat(sub_bands, dim=1)
        return F.conv_transpose2d(x_stacked, self.inv_haar_kernel, stride=2, groups=sub_bands[0].shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bypass processing if spatial dimensions are too small for a 2x2 wavelet operation
        if x.shape[2] < 4 or x.shape[3] < 4:
            return x

        residual = x
        sub_bands = self.dwt(x)

        # Process each frequency sub-band (LL, LH, HL, HH)
        restored_bands = []
        for i in range(4):
            compressed = self.compressors[i](sub_bands[i])
            enhanced = self.process_block(compressed)
            restored = self.expanders[i](enhanced)
            restored_bands.append(restored)

        # Reconstruct spatial features
        reconstructed = self.idwt(tuple(restored_bands))
        
        # Apply learnable residual weighting and normalize
        out = residual + (self.alpha * reconstructed)
        return self.out_act(self.out_norm(out))