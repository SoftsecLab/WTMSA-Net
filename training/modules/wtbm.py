import torch
import torch.nn as nn

class WTBM(nn.Module):
    def __init__(self):
        super(WTBM, self).__init__()
        self.fc_low = None  
        self.fc_reduce = nn.Linear(114688, 1792) 
        self.mlp = nn.Sequential(
            nn.Linear(1792, 1792),  
            nn.ReLU(),
            nn.Linear(1792, 1792)   
        )
        self.fc_rescale = nn.Linear(1792, 1792)  

    def dwt(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        ll = (x[:, :, 0::2, 0::2] + x[:, :, 1::2, 0::2] + x[:, :, 0::2, 1::2] + x[:, :, 1::2, 1::2]) / 4
        lh = (-x[:, :, 0::2, 0::2] - x[:, :, 1::2, 0::2] + x[:, :, 0::2, 1::2] + x[:, :, 1::2, 1::2]) / 4
        hl = (-x[:, :, 0::2, 0::2] + x[:, :, 1::2, 0::2] - x[:, :, 0::2, 1::2] + x[:, :, 1::2, 1::2]) / 4
        hh = (x[:, :, 0::2, 0::2] - x[:, :, 1::2, 0::2] - x[:, :, 0::2, 1::2] + x[:, :, 1::2, 1::2]) / 4

        high = torch.cat([lh, hl, hh], dim=1)
        return ll, high

    def idwt(self, F_low, F_high):
        B, C, H, W = F_low.shape
        lh, hl, hh = torch.chunk(F_high, 3, dim=1)
        out = torch.zeros(B, C, H * 2, W * 2, device=F_low.device, dtype=F_low.dtype)

        out[:, :, 0::2, 0::2] = F_low - lh - hl + hh
        out[:, :, 1::2, 0::2] = F_low - lh + hl - hh
        out[:, :, 0::2, 1::2] = F_low + lh - hl - hh
        out[:, :, 1::2, 1::2] = F_low + lh + hl + hh

        return out

    def forward(self, x):
        device = x.device  
        B, C, H, W = x.shape
        F_low, F_high = self.dwt(x)


        feature_dim = F_low.view(B, -1).shape[1]

   
        if self.fc_low is None:  
            self.fc_low = nn.Linear(feature_dim, feature_dim)
            self.fc_low = self.fc_low.to(device)  


        F_low_flat = F_low.view(B, -1)
        F_low_proc = torch.relu(self.fc_low(F_low_flat))
        F_low_proc = F_low_proc.view(B, C, H // 2, W // 2)

        F_high_proc = torch.sigmoid(F_high)
        F_recon = self.idwt(F_low_proc, F_high_proc)
        F_recon_flat = F_recon.view(B, -1)


        F_recon_reduced = torch.relu(self.fc_reduce(F_recon_flat))  


        F_W = self.mlp(F_recon_reduced)
        F_W = F_W.unsqueeze(2).unsqueeze(3).expand(-1, -1, 8, 8)
        return F_W