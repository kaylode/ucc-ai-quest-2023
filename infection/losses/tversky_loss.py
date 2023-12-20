import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    def __init__(self, smooth=1, alpha=0.7, gamma=0.75, **kwargs):
        super(FocalTverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = 1-alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        prediction = F.softmax(logits, dim=1)  
        if len(targets.shape) == 3:
            num_classes = prediction.shape[1]
            targets = F.one_hot(
                targets.long(), num_classes=num_classes
            ).permute(0, 3, 1, 2)
        
        #flatten label and prediction tensors
        prediction = prediction.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (prediction * targets).sum()    
        FP = ((1-targets) * prediction).sum()
        FN = (targets * (1-prediction)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha*FN + self.beta*FP + self.smooth)  
        loss = (1 - tversky)**self.gamma
        
        loss_dict = {"FT": loss.item()}
        return loss, loss_dict