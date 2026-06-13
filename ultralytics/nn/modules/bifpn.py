import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, C2f
from ultralytics.utils.torch_utils import fuse_conv_and_bn

class WeightedFusion(nn.Module):
    """Fast Normalized Fusion"""
    def __init__(self, num_inputs):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs))
        self.eps = 1e-4
        
    def forward(self, x):
        # x: list of tensors
        w = F.relu(self.weights)
        w = w / (w.sum() + self.eps)
        return sum(w[i] * x[i] for i in range(len(x)))

class BiFPNBlock(nn.Module):
    """
    BiFPN basic unit: depthwise separable convolution + weighted fusion
    In YAML: args: [channels, shortcut=True/False]
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(self.c, self.c, 1, 1)
        self.cv3 = Conv(self.c, c2, 1, 1)
        
        # Depthwise separable convolution (mimics BiFPN lightweight conv)
        self.dwconv = nn.Sequential(
            nn.Conv2d(self.c, self.c, 3, 1, 1, groups=self.c, bias=False),
            nn.BatchNorm2d(self.c),
            nn.SiLU(),
            nn.Conv2d(self.c, self.c, 1, bias=False),
            nn.BatchNorm2d(self.c),
            nn.SiLU()
        )
        
        self.fusion = WeightedFusion(2)  # 2 inputs
        self.add = shortcut and c1 == c2
        
    def forward(self, x):
        # If x is a single tensor, process directly; if list, fuse first
        if isinstance(x, list):
            if len(x) == 2:
                x = self.fusion(x)
            else:
                x = torch.cat(x, dim=1)
        
        residual = x
        x = self.cv1(x)
        x = self.dwconv(x)
        x = self.cv2(x)
        if self.add:
            x = x + residual
        return self.cv3(x)

class BiFPN(nn.Module):
    """Complete BiFPN module (standby, can be used directly in models)"""
    def __init__(self, ch_list, out_ch=256):
        super().__init__()
        # ch_list: [C3_channels, C4_channels, C5_channels]
        self.ch_list = ch_list
        
        # Input projection
        self.proj = nn.ModuleList([
            Conv(ch, out_ch, 1) for ch in ch_list
        ])
        
        # Bidirectional fusion blocks (extensible to multiple layers)
        self.bifpn_td = nn.ModuleList()  # top-down
        self.bifpn_bu = nn.ModuleList()  # bottom-up
        
        for i in range(len(ch_list) - 1):
            self.bifpn_td.append(BiFPNBlock(out_ch, out_ch))
            self.bifpn_bu.append(BiFPNBlock(out_ch, out_ch))
            
    def forward(self, features):
        # features: [P3, P4, P5]
        feats = [proj(f) for proj, f in zip(self.proj, features)]
        
        # Top-down
        td_feats = [feats[-1]]
        for i in range(len(feats) - 2, -1, -1):
            up = F.interpolate(td_feats[-1], size=feats[i].shape[2:], mode='nearest')
            fused = self.bifpn_td[len(feats) - 2 - i]([up, feats[i]])
            td_feats.append(fused)
        td_feats = td_feats[::-1]
        
        # Bottom-up
        bu_feats = [td_feats[0]]
        for i in range(len(td_feats) - 1):
            down = F.max_pool2d(bu_feats[-1], 3, stride=2, padding=1)
            fused = self.bifpn_bu[i]([down, td_feats[i + 1]])
            bu_feats.append(fused)
            
        return bu_feats

class P2Downsample(nn.Module):
    """
    Lightweight P2 downsampling: uses spatial reduction instead of standard convolution
    Computation cost is ~8% of Conv3x3
    """
    def __init__(self, in_ch, out_ch, method='avgpool'):
        super().__init__()
        self.method = method
        
        if method == 'avgpool':
            # Method 1: Avg pooling + 1x1 projection (recommended, lightweight)
            self.pool = nn.AvgPool2d(2, stride=2)
            self.proj = Conv(in_ch, out_ch, 1, 1)  # 1x1 conv only for channel adjustment
            
        elif method == 'slice':
            # Method 2: Spatial slicing + concat (parameter-free, max speed)
            self.proj = Conv(in_ch * 4, out_ch, 1, 1)  # channels x4 after concat 4 slices
            
        elif method == 'pixelshuffle_inv':
            # Method 3: Inverse pixel shuffle (smoother gradient flow)
            self.proj = Conv(in_ch, out_ch * 4, 1, 1)
            self.pixel_unshuffle = nn.PixelUnshuffle(2)  # spatial halved, channels x4
            
        else:  # 'depthwise'
            # Method 4: Depthwise separable convolution (75% lighter than standard Conv)
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, 2, 1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU()
            )
    
    def forward(self, x):
        if self.method == 'avgpool':
            x = self.pool(x)
            x = self.proj(x)
            
        elif self.method == 'slice':
            # Slice 160x160 feature map into 4 80x80 patches, concat channels then project
            # Similar to PixelUnshuffle but implemented with slicing
            b, c, h, w = x.shape
            x = x.view(b, c, h // 2, 2, w // 2, 2)
            x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
            x = x.view(b, c * 4, h // 2, w // 2)
            x = self.proj(x)
            
        elif method == 'pixelshuffle_inv':
            x = self.proj(x)  # expand channels first
            x = self.pixel_unshuffle(x)  # halve spatial dimensions
            
        else:  # depthwise
            x = self.down(x)
            
        return x

class LightweightBiFPNBlock(nn.Module):
    """
    Lightweight BiFPN block: uses special downsampling for P2 layer, standard ops for others
    """
    def __init__(self, channels, num_levels=4, p2_down_method='avgpool'):
        super().__init__()
        self.num_levels = num_levels  # 4: P2, P3, P4, P5
        self.channels = channels
        
        # Top-Down: uniform upsampling + convolution
        self.conv_td = nn.ModuleList([
            Conv(channels, channels, 3, 1) for _ in range(num_levels - 1)
        ])
        
        # Bottom-Up:
        # P2->P3 uses lightweight downsampling (index 0)
        # P3->P4, P4->P5 uses standard convolution (index 1,2)
        self.downsamples = nn.ModuleList()
        
        # P2->P3: lightweight downsampling
        self.downsamples.append(P2Downsample(channels, channels, method=p2_down_method))
        
        # P3->P4, P4->P5: standard stride=2 convolution (or depthwise separable)
        for i in range(num_levels - 2):
            self.downsamples.append(Conv(channels, channels, 3, 2))  # stride=2
            
        # Post-fusion convolution
        self.conv_bu = nn.ModuleList([
            Conv(channels, channels, 1, 1) for _ in range(num_levels - 1)
        ])
        
        # Weighted fusion weights
        self.w_td = nn.Parameter(torch.ones(2))  # 2 inputs: upsampled + lateral
        self.w_bu = nn.Parameter(torch.ones(2))  # 2 inputs: downsampled + lateral
        
    def forward(self, inputs):
        """
        inputs: [P2, P3, P4, P5] from high to low resolution
        P2: 160x160, P3: 80x80, P4: 40x40, P5: 20x20 (640 input)
        """
        # ===== Top-Down path =====
        td_feats = [inputs[-1]]  # start from P5
        
        for i in range(self.num_levels - 2, -1, -1):  # i=2,1,0 -> P4,P3,P2
            feat_high = F.interpolate(td_feats[-1], size=inputs[i].shape[2:], 
                                     mode='nearest')
            # Weighted fusion
            w = F.relu(self.w_td)
            w = w / (w.sum() + 1e-4)
            fused = w[0] * feat_high + w[1] * inputs[i]
            fused = self.conv_td[self.num_levels - 2 - i](fused)
            td_feats.append(fused)
        
        td_feats = td_feats[::-1]  # [P2, P3, P4, P5]
        
        # ===== Bottom-Up path =====
        bu_feats = [td_feats[0]]  # P2
        
        for i in range(self.num_levels - 1):  # i=0,1,2 -> P2->P3, P3->P4, P4->P5
            # Use corresponding downsampler (P2->P3: lightweight, others: standard)
            feat_low = self.downsamples[i](bu_feats[-1])
            
            # Ensure size match
            target_size = td_feats[i + 1].shape[2:]
            if feat_low.shape[2:] != target_size:
                feat_low = F.interpolate(feat_low, size=target_size, mode='nearest')
            
            # Weighted fusion
            w = F.relu(self.w_bu)
            w = w / (w.sum() + 1e-4)
            fused = w[0] * feat_low + w[1] * td_feats[i + 1]
            fused = self.conv_bu[i](fused)
            bu_feats.append(fused)
            
        return bu_feats

class LightweightBiFPN(nn.Module):
    """
    Complete Lightweight BiFPN: stacks multiple blocks, P2 always uses lightweight downsampling
    """
    def __init__(self, ch_list=[128, 256, 256, 256], out_ch=256, 
                 num_blocks=2, p2_method='avgpool'):
        """
        ch_list: [P2_ch, P3_ch, P4_ch, P5_ch] input channels before projection
        """
        super().__init__()
        self.num_levels = len(ch_list)
        
        # Input projection (P2 keeps smaller channels like 128; others 256)
        self.proj = nn.ModuleList([
            Conv(ch_list[i], out_ch, 1) 
            for i in range(self.num_levels)
        ])
        
        # Stack BiFPN blocks
        self.blocks = nn.ModuleList([
            LightweightBiFPNBlock(
                channels=out_ch, 
                num_levels=self.num_levels,
                p2_down_method=p2_method if i == 0 else 'depthwise'  # first block lightweight, subsequent blocks standard
            ) for i in range(num_blocks)
        ])
        
    def forward(self, features):
        # Project to uniform channels
        feats = [proj(f) for proj, f in zip(self.proj, features)]
        
        # Pass through BiFPN blocks
        for block in self.blocks:
            feats = block(feats)
            
        return feats

class BiFPN_P2(nn.Module):
    """
    Ultralytics-compatible BiFPN module (P2 version)
    Usage in YAML: [-1, 1, BiFPN_P2, [[128, 256, 256, 256], 256, 2]]
    args: [ch_list, out_ch, num_blocks]
    """
    def __init__(self, ch_list=[128, 256, 256, 256], out_ch=256, num_blocks=2):
        super().__init__()
        self.bifpn = LightweightBiFPN(ch_list, out_ch, num_blocks, p2_method='avgpool')
        
    def forward(self, x):
        return self.bifpn(x)