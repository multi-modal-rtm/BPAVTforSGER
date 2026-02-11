import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from scipy.special import softmax
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore', category=UserWarning)

LABELS = ['Boredom', 'Engagement', 'Confusion', 'Frustration ']

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    gamma > 0 reduces the relative loss for well-classified examples (p > .5), 
    putting more focus on hard, misclassified examples.
    """
    def __init__(self, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def calculate_classification_metrics(logits, labels):
    """
    Calculates multiclass classification metrics for each emotion.
    """
    preds = np.argmax(logits, axis=2)
    metrics = {}
    
    for i, emotion in enumerate(LABELS):
        emotion_name = emotion.strip().lower()
        emotion_preds = preds[:, i]
        emotion_labels = labels[:, i]
        emotion_logits = logits[:, i, :]
        
        emotion_probs = softmax(emotion_logits, axis=1)
        try:
            auc_score = roc_auc_score(emotion_labels, emotion_probs, multi_class='ovr', average='macro')
        except ValueError:
            auc_score = 0.0
        metrics[f'auc_macro_{emotion_name}'] = auc_score

        accuracy = accuracy_score(emotion_labels, emotion_preds)
        f1 = f1_score(emotion_labels, emotion_preds, average='macro', zero_division=0)
        
        metrics[f'accuracy_{emotion_name}'] = accuracy
        metrics[f'f1_macro_{emotion_name}'] = f1

    exact_match_accuracy = np.mean(np.all(preds == labels, axis=1))
    metrics['exact_match_accuracy'] = exact_match_accuracy
    
    print("\n--- Classification Report per Emotion ---")
    for i, emotion in enumerate(LABELS):
        print(f"\n--- {emotion.strip()} ---")
        try:
            report = classification_report(labels[:, i], preds[:, i], zero_division=0)
            print(report)
        except Exception as e:
            print(f"Could not generate report: {e}")
            
    return metrics