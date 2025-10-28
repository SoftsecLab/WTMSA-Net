import torch
import torch.nn as nn
import torch.nn.functional as F


class SimAM(nn.Module):
    """Simple Attention Module (SimAM) — parameter-free attention"""

    def __init__(self, lam=1e-4, epsilon=1e-5):
        super(SimAM, self).__init__()
        self.lam = lam
        self.epsilon = epsilon

    def forward(self, x):
        # x shape: (B, C, H, W)
        n = x.size(2) * x.size(3) - 1  # number of pixels excluding current one per channel

        mean = (x.sum(dim=(2, 3), keepdim=True) - x) / n  # mean excluding current pixel
        var = ((x - mean) ** 2).sum(dim=(2, 3), keepdim=True) / n  # variance excluding current pixel

        energy = (x - mean) ** 2 + self.lam * var
        attention = 1.0 / (4 * (energy + self.epsilon))

        return attention * x


class WTConv(nn.Module):
    """
    Placeholder Wavelet Transform Convolution module.
    In practice, replace with real WTConv operation or implementation.
    Here implemented as a conv layer for example.
    """

    def __init__(self, in_channels):
        super(WTConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class LFAM(nn.Module):
    def __init__(self, channels):
        super(LFAM, self).__init__()
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, channels, bias=False),
        )

        self.wtconv = WTConv(channels)
        self.simam = SimAM()

        self.conv1x1_edge = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, F_W):
        
        Fi = self.conv3x3(F_W)
        Fi = self.bn(Fi)
        Fi = F.relu(Fi)

    # Path 1: channel attention via GAP + FC + ReLU
        gap = F.adaptive_avg_pool2d(Fi, (1, 1)).view(Fi.size(0), -1)  
        Fl1 = self.fc(gap)
        Fl1 = F.relu(Fl1)

    # Reshape Fl1 to (B, C, 1, 1) before expanding
        Fl1 = Fl1.view(Fi.size(0), Fi.size(1), 1, 1)  

    # Expand Fl1 to match Fi's spatial dimensions
        Fl1 = Fl1.expand_as(Fi)  
        Fl1 = Fi * Fl1  # channel-wise scaling

    # Path 2: WTConv + SimAM + edge enhancement
        Fwt = self.wtconv(Fi)
        Fs1 = self.simam(Fwt)

    # Edge features extraction
        Fs1_ap = F.avg_pool2d(Fs1, kernel_size=3, stride=1, padding=1)
        Fs1_sub = Fs1 - Fs1_ap
        Fs1_edge = self.sigmoid(self.conv1x1_edge(Fs1_sub)) * Fs1 + Fs1

    # Apply SimAM attention again
        Fl2 = self.simam(Fs1_edge)

        # Combine two paths
        FL = Fl1 + Fl2

        return FL
