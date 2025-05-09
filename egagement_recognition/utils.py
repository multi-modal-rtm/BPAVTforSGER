import os
import shutil
import torch
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

def save_checkpoint(state: dict, is_best: bool, ckpt_dir: str):
    """
    Saves the current training state to 'last.pth';
    if is_best is True, also copies it to 'best.pth'.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    last_path = os.path.join(ckpt_dir, 'last.pth')
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(ckpt_dir, 'best.pth')
        shutil.copyfile(last_path, best_path)

def load_checkpoint(ckpt_path: str, model, optimizer=None, scaler=None):
    """
    Loads state from ckpt_path into model (and optimizer/scaler if provided).
    Returns the full checkpoint dict for inspection.
    """
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model_state = checkpoint.get('model_state', checkpoint)
    model.load_state_dict(model_state)
    if optimizer is not None and 'optim_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optim_state'])
    if scaler is not None and 'scaler_state' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state'])
    return checkpoint

class AverageMeter:
    """
    Tracks and updates the current value, sum, count, and average of a metric.
    Useful for smoothing losses or accuracies across batches.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0

def compute_mAP(y_true, y_scores, num_classes):
    """
    Compute mean Average Precision (mAP) across classes.
    
    Args:
        y_true (array-like, shape [n_samples]): integer ground-truth labels (0 ... num_classes-1)
        y_scores (array-like, shape [n_samples, num_classes]): predicted scores or probabilities
        num_classes (int): number of classes
    
    Returns:
        mAP (float): mean of per-class average precision
        ap_per_class (list of float): list of AP for each class
    """
    # Binarize the ground-truth labels
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    
    ap_per_class = []
    for cls in range(num_classes):
        ap = average_precision_score(y_true_bin[:, cls], y_scores[:, cls])
        ap_per_class.append(ap)
    
    mAP = float(np.mean(ap_per_class))
    return mAP, ap_per_class
