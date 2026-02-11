import torch
import torch.nn as nn
from omegaconf import DictConfig
from focal_loss.focal_loss import FocalLoss

class FocalLossWrapper(nn.Module):
    """
    A wrapper for the focal-loss-torch library to handle raw logits
    from the model by applying a softmax activation first.
    """
    def __init__(self, weights=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.focal_loss = FocalLoss(weights=weights, gamma=gamma, reduction=reduction)

    def forward(self, inputs, targets):
        """
        Applies softmax to the inputs before passing them to the focal loss.
        Args:
            inputs (torch.Tensor): The raw logits from the model.
            targets (torch.Tensor): The ground truth labels.
        """
        probs = torch.nn.functional.softmax(inputs, dim=1)

        return self.focal_loss(probs, targets)

def get_criterion(config: DictConfig, device: torch.device) -> nn.Module:
    """
    Creates the appropriate loss function based on the config.
    """
    loss_name = config.loss.name
    
    if loss_name == "FocalLoss":
        weights = config.loss.get("focal_loss_weights", None)
        gamma = config.loss.get("focal_loss_gamma", 2.0)

        if weights:
            weights = torch.tensor(weights, dtype=torch.float32).to(device)
            
        return FocalLossWrapper(weights=weights, gamma=gamma, reduction='mean')
        
    elif loss_name == "CrossEntropyLoss":
        weights = config.loss.get("class_weights", None)
        if weights:
            weights = torch.tensor(weights, dtype=torch.float32).to(device)
        return nn.CrossEntropyLoss(weight=weights)
        
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")