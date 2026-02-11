import torch
import torch.nn as nn
# Rename the imported class to avoid a name conflict with our new wrapper
from focal_loss.focal_loss import FocalLoss as FocalLossLib
from omegaconf import DictConfig

class FocalLoss(nn.Module):
    """
    Wrapper around the focal_loss_torch library that applies a softmax to the
    model's raw outputs (logits) before calculating the loss.
    This makes it a drop-in replacement for nn.CrossEntropyLoss.
    """
    def __init__(self, weights, gamma, reduction='mean'):
        super(FocalLoss, self).__init__()
        # The library's FocalLoss expects probabilities, so we create an instance here
        self.focal_loss = FocalLossLib(weights=weights, gamma=gamma, reduction=reduction)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): The raw, unnormalized outputs from the model (logits).
            targets (torch.Tensor): The ground truth labels.
        """
        # Apply softmax to the raw logits to get probabilities
        probs = nn.functional.softmax(inputs, dim=1)
        return self.focal_loss(probs, targets)


def get_criterion(config: DictConfig, device: torch.device) -> nn.Module:
    """
    Creates the loss function (criterion) based on the config.
    Supports Focal Loss and standard CrossEntropyLoss.
    """
    loss_name = config.loss.name

    if loss_name == "FocalLoss":
        # The library expects the argument 'weights', not 'alpha'.
        weights = torch.tensor(config.loss.class_weights, dtype=torch.float32).to(device)
        gamma = config.loss.gamma
        
        # Return our new, safe wrapper class
        return FocalLoss(weights=weights, gamma=gamma, reduction='mean')
    else: # Default CrossEntropyLoss
        class_weights = torch.tensor(config.loss.class_weights, dtype=torch.float32).to(device)
        return nn.CrossEntropyLoss(weight=class_weights)









# import torch
# import torch.nn as nn
# # Rename the imported class to avoid a name conflict with our new wrapper
# from focal_loss.focal_loss import FocalLoss as FocalLossLib
# from omegaconf import DictConfig

# class FocalLoss(nn.Module):
#     """
#     Wrapper around the focal_loss_torch library that applies a softmax to the
#     model's raw outputs (logits) before calculating the loss.
#     This makes it a drop-in replacement for nn.CrossEntropyLoss.
#     """
#     def __init__(self, weights, gamma, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         # The library's FocalLoss expects probabilities, so we create an instance here
#         self.focal_loss = FocalLossLib(weights=weights, gamma=gamma, reduction=reduction)

#     def forward(self, inputs, targets):
#         """
#         Args:
#             inputs (torch.Tensor): The raw, unnormalized outputs from the model (logits).
#             targets (torch.Tensor): The ground truth labels.
#         """
#         # Apply softmax to the raw logits to get probabilities
#         probs = nn.functional.softmax(inputs, dim=1)
#         return self.focal_loss(probs, targets)


# def get_criterion(config: DictConfig, device: torch.device) -> nn.Module:
#     """
#     Creates the loss function (criterion) based on the config.
#     Supports Focal Loss and standard CrossEntropyLoss.
#     """
#     loss_name = config.get("loss_function", "CrossEntropyLoss") # Default to CE

#     if loss_name == "FocalLoss":
#         # The library expects the argument 'weights', not 'alpha'.
#         weights = torch.tensor(config.class_weights, dtype=torch.float32).to(device)
#         gamma = config.get("focal_loss_gamma", 2.0)
        
#         # Return our new, safe wrapper class
#         return FocalLoss(weights=weights, gamma=gamma, reduction='mean')
#     else: # Default CrossEntropyLoss
#         class_weights = torch.tensor(config.class_weights, dtype=torch.float32).to(device)
#         return nn.CrossEntropyLoss(weight=class_weights)

