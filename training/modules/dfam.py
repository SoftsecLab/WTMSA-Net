import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pw_conv(x)
        x = self.bn2(x)
        return x

class DFAM(nn.Module):
    def __init__(self, in_channels_wtbm, in_channels_baseline):
        super(DFAM, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels_wtbm + in_channels_baseline, (in_channels_wtbm + in_channels_baseline) // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d((in_channels_wtbm + in_channels_baseline) // 2),
            nn.ReLU(inplace=True)
        )
        self.dw_sep_conv = DepthwiseSeparableConv((in_channels_wtbm + in_channels_baseline) // 2, (in_channels_wtbm + in_channels_baseline) // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, F_W, F):
        
        concat = torch.cat([F_W, F], dim=1)  # (B, C1 + 1792, H, W)
        
        F_a = self.reduce_conv(concat) + F_W

        F_b = self.dw_sep_conv(F_a)
        F_b = self.relu(F_b + F_a)
        return F_b