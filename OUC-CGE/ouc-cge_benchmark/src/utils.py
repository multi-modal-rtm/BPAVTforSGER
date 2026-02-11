import torch
import os
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import numpy as np
from sklearn.preprocessing import label_binarize

def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Saves the current state of training to a file."""
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Loads a checkpoint and returns the epoch number to start from."""
    if os.path.exists(checkpoint_path):
        print(f"=> Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"=> Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']
    else:
        print(f"=> No checkpoint found at '{checkpoint_path}'")
        return 0

def calculate_metrics(labels, preds, probs, num_classes=3, class_names=None):
    """Calculates and prints all required metrics."""

    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
    labels_binarized = label_binarize(labels, classes=range(num_classes))
    
    if labels_binarized.shape[1] < num_classes:

        padded_labels = np.zeros((labels_binarized.shape[0], num_classes))
        padded_labels[:, :labels_binarized.shape[1]] = labels_binarized
        labels_binarized = padded_labels
        
        padded_probs = np.zeros((probs.shape[0], num_classes))
        padded_probs[:, :probs.shape[1]] = probs
        probs = padded_probs

    try:

        if len(np.unique(labels)) < num_classes:
             macro_roc_auc = 0.0 
             print("Warning: Not all classes were present in this evaluation set. ROC AUC is set to 0.0.")
        else:
             macro_roc_auc = roc_auc_score(labels_binarized, probs, average='macro', multi_class='ovr')
    except ValueError as e:
        print(f"Could not compute ROC AUC: {e}")
        macro_roc_auc = 0.0

    print("\n--- Evaluation Metrics ---")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print(f"Macro ROC AUC: {macro_roc_auc:.4f}")
    print("\n--- Classification Report ---")
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))
    
    metrics_dict = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'macro_roc_auc': macro_roc_auc
    }
    return metrics_dict