from infection.losses.dice_loss import DiceLoss
from infection.losses.ce_loss import CELoss
from infection.losses.lovasz_loss import LovaszSoftmax
from infection.losses.tversky_loss import FocalTverskyLoss
from typing import Dict

def get_loss(config:Dict):
    name = config.get("loss_name", 'cross_entropy')
    if name == 'dice':
        return DiceLoss(**config)
    
    elif name == 'cross_entropy':
        return CELoss(**config)
    
    elif name == 'lovasz':
        return LovaszSoftmax(**config)
    
    elif name == 'tversky':
        return FocalTverskyLoss(**config)
    
    else:
        raise NotImplementedError(f"loss {name} is not implemented")