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
        loss_dict = {"CE": loss.item()}
        return loss, loss_dict


class OhemCELoss(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: List = None, thresh: float = 0.7, **kwargs) -> None:
        super().__init__()

        self.weight = weight
        if self.weight is not None:
            self.weight = torch.FloatTensor(self.weight)

        self.ignore_label = ignore_label
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=ignore_label, reduction='none')

    def forward(self, logits, targets) -> torch.Tensor:

        if self.weight is not None:
            self.criterion.weight = self.criterion.weight.to(logits.device)

        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        n_min = targets[targets != self.ignore_label].numel() // 16

        loss = self.criterion(logits, targets).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        
        loss = loss_hard.mean()
        
        loss_dict = {"OhemCE": loss.item()}
        return loss, loss_dict