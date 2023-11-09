from typing import Dict, List
import torch
from torch import nn

class CELoss(nn.Module):
    r"""CELoss is warper of cross-entropy loss"""

    def __init__(self, label_smoothing:float=0.0, **kwargs):
        super(CELoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        loss = nn.functional.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
        return loss