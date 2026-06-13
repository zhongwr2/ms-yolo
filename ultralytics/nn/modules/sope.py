import torch
import torch.nn as nn

class AffineTransform(nn.Module):
    """
    Affine transform layer in SOPE
    Replaces LayerNorm, simpler and more efficient
    """
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))  # learnable scaling factor
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))   # learnable bias
    
    def forward(self, x):
        # Simple linear transform: y = alpha * x + beta
        return self.alpha * x + self.beta

# Complete SOPE module example
class SOPE(nn.Module):
    def __init__(self, in_chans=3, embed_dim=128, patch_size=4):
        super().__init__()
        # Pre-affine transform
        self.pre_affine = AffineTransform(in_chans)
        
        # Overlapping convolution sequence (gradual downsampling to 1/4)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim//2, 3, stride=2, padding=1),  # H/2
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, 3, stride=2, padding=1), # H/4
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        
        # Post-affine transform
        self.post_affine = AffineTransform(embed_dim)
    
    def forward(self, x):
        # Pre-affine transform
        x = self.pre_affine(x)
        
        # Overlapping convolution downsampling
        x = self.patch_embed(x)
        
        # Post-affine transform
        x = self.post_affine(x)
        
        return x