__author__      = "Grzegorz Baumann"
__contact__     = "g.baumann@unibas.ch"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F 

# ==========================================
# VQ-WAVE NETWORK (Hybrid Head + No Stride)
# ==========================================
class SEBlock1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class InceptionBlock1D_SE(nn.Module):
    def __init__(self, in_channels, out_channels, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        b_ch = out_channels // 4
        self.b1 = nn.Conv1d(in_channels, b_ch, kernel_size=3, padding=1)
        self.b2 = nn.Conv1d(in_channels, b_ch, kernel_size=11, padding=5)
        self.b3 = nn.Conv1d(in_channels, b_ch, kernel_size=21, padding=10)
        self.b4 = nn.Conv1d(in_channels, b_ch, kernel_size=41, padding=20)
        self.mix = nn.Conv1d(b_ch * 4, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.se = SEBlock1D(out_channels)
        self.relu = nn.ReLU()
        if self.use_residual:
            self.sc = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
                      if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        out = self.mix(out)
        out = self.bn(out)
        out = self.se(out)
        if self.use_residual: out += self.sc(x)
        return self.relu(out)

class VQWaveNet(nn.Module):
    def __init__(self, width_factor=1.5, in_channels=4):
        super().__init__()
        base = int(32 * width_factor)
        
        self.entry = nn.Sequential(
            nn.Conv1d(in_channels, base, kernel_size=7, padding=3),
            nn.BatchNorm1d(base), nn.ReLU(),
            nn.Conv1d(base, base, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(base), nn.ReLU()
        )
        
        self.block1 = InceptionBlock1D_SE(base, base*2)
        self.block2 = InceptionBlock1D_SE(base*2, base*4)
        self.block3 = InceptionBlock1D_SE(base*4, base*4)
        self.block4 = InceptionBlock1D_SE(base*4, base*8) # 384 Ch
        
        # Hybrid Head: Avg (Amp/Freq) + Center Slice (Phase)
        self.head = nn.Sequential(
            nn.Linear(base*8 * 2, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 8) 
        )

    def forward(self, x):
        x = self.entry(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # HYBRID POOLING
        # Average helps with Noise
        x_avg = x.mean(dim=2)
        
        # Center Slice for improved phase estimation
        mid_idx = x.shape[2] // 2
        x_ctr = x[:, :, mid_idx]
        
        x_combined = torch.cat([x_avg, x_ctr], dim=1)
        return self.head(x_combined)    
