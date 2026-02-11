"""
Utility functions for the DAiSEE benchmark, including data transforms,
loss functions, and metric calculations.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_data_transforms(frame_size=224):
    """
    Returns the standardized data transforms for video frames.
    """
    return transforms.Compose([
        transforms.Resize((frame_size, frame_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss.
    
    This loss function is designed to address class imbalance by
    down-weighting the loss assigned to well-classified examples.
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C) tensor of raw logits from the model.
            targets: (B,) tensor of ground-truth class indices.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
 
        pt = torch.exp(-ce_loss)

        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

LABELS = ['Boredom', 'Engagement', 'Confusion', 'Frustration']

def calculate_classification_metrics(all_logits, all_labels):
    """
    Calculates classification metrics for all four affective tasks.

    Args:
        all_logits (list of np.array): A list of 4 arrays, where each array
                                       has shape (N_samples, 4 classes)
                                       containing the logits for a task.
        all_labels (np.array): An array of shape (N_samples, 4) containing
                               the ground-truth labels.
                               
    Returns:
        dict: A dictionary containing accuracy and F1 scores for each
              task and the averaged scores.
    """
    metrics = {}
    all_preds = []
    

    for i in range(len(LABELS)):
        preds = np.argmax(all_logits[i], axis=1)
        all_preds.append(preds)

    all_preds_np = np.stack(all_preds, axis=1)
    
    task_accuracies = []
    task_f1_macros = []

    for i, label_name in enumerate(LABELS):
        task_labels = all_labels[:, i]
        task_preds = all_preds_np[:, i]

        acc = accuracy_score(task_labels, task_preds)
        metrics[f'accuracy_{label_name.lower()}'] = acc
        task_accuracies.append(acc)

        f1 = f1_score(task_labels, task_preds, average='macro', zero_division=0)
        metrics[f'f1_macro_{label_name.lower()}'] = f1
        task_f1_macros.append(f1)

    metrics['accuracy_avg'] = np.mean(task_accuracies)
    metrics['f1_macro_avg'] = np.mean(task_f1_macros)
    
    return metrics