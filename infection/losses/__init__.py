from typing import Any, Dict, Iterable
from infection.losses.dice_loss import DiceLoss
from infection.losses.ce_loss import CELoss, OhemCELoss
from infection.losses.lovasz_loss import LovaszSoftmax
from infection.losses.tversky_loss import FocalTverskyLoss
from typing import Dict

import torch
import torch.nn as nn

class MultiLoss(nn.Module):
    """Wrapper class for combining multiple loss function"""

    def __init__(self, losses: Iterable[nn.Module], weights=None, **kwargs):
        super().__init__()
        self.losses = losses
        self.weights = [1.0 for _ in range(len(losses))] if weights is None else weights

    def forward(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        device: torch.device = None,
    ):
        """
        Forward inputs and targets through multiple losses
        """
        total_loss = 0
        total_loss_dict = {}

        for weight, loss_fn in zip(self.weights, self.losses):
            loss, loss_dict = loss_fn(outputs, batch)
            total_loss += weight * loss
            total_loss_dict.update(loss_dict)

        total_loss_dict.update({"Total": total_loss.item()})
        return total_loss, total_loss_dict



def get_loss(config:Dict):

    losses = []
    for loss_cfg in config:
        name = loss_cfg["name"]
        loss_args = loss_cfg.get("args", {})

        if name == 'dice':
            loss = DiceLoss(**loss_args)
        
        elif name == 'cross_entropy':
            loss = CELoss(**loss_args)
        
        elif name == 'ohem_cross_entropy':
            loss = OhemCELoss(**loss_args)
        
        elif name == 'lovasz':
            loss = LovaszSoftmax(**loss_args)
        
        elif name == 'tversky':
            loss = FocalTverskyLoss(**loss_args)
        
        else:
            raise NotImplementedError(f"loss {name} is not implemented")

        losses.append(loss)
        
    return MultiLoss(losses)