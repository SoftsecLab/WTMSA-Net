import torch.nn as nn
from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC
import torch

@LOSSFUNC.register_module(module_name="totalloss")
class totalloss(AbstractLossClass):


    def __init__(self, lambda_fc=0.5, lambda_da=0.3, lambda_bce=0.8, da_alpha=0.25, da_gamma=2.0):
        super().__init__()
        self.lambda_fc = lambda_fc  # λ1
        self.lambda_da = lambda_da  # λ2
        self.lambda_bce = lambda_bce  # λ3
        self.da_alpha = da_alpha
        self.da_gamma = da_gamma

        # Feature Classification Loss (no softmax, no w,b)
        self.fcl_loss_fn = nn.CrossEntropyLoss()

    # Difficulty-Aware Loss
    def daloss(self, prob: torch.Tensor, labels: torch.Tensor):
        eps = 1e-8
        prob = torch.clamp(prob, eps, 1 - eps)
        L_pos = -self.da_alpha * ((1 - prob) ** self.da_gamma) * labels * torch.log(prob)
        L_neg = -(1 - self.da_alpha) * (prob ** self.da_gamma) * (1 - labels) * torch.log(1 - prob)
        return (L_pos + L_neg).mean()

    def forward(self, logits: torch.Tensor, prob: torch.Tensor, labels: torch.Tensor):
        # Feature Classification Loss (no softmax)
        loss_fc = self.lambda_fc * self.fcl_loss_fn(logits, labels)

        # Difficulty-Aware Loss
        loss_da = self.lambda_da * self.daloss(prob, labels)

        # Binary Cross-Entropy Loss
        loss_bce = self.lambda_bce * F.binary_cross_entropy(prob, labels.float())

        # Overall Loss
        total_loss = loss_fc + loss_da + loss_bce

        return total_loss