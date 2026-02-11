import torch
import os
import shutil
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report
)

def load_config(config_path):
    """Loads a YAML config file."""
    import yaml
    from omegaconf import OmegaConf
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return OmegaConf.create(config_dict)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit(1)

def save_checkpoint(state, filename="checkpoint.pth"):
    """Saves model and optimizer state."""
    torch.save(state, filename)

def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device="cpu"):
    """Loads model and optimizer state from a checkpoint."""
    if not os.path.exists(filepath):
        print("No checkpoint found. Starting from scratch.")
        return 0, 0.0, 0 
    
    try:
        checkpoint = torch.load(filepath, map_location=device)

        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint) 

        if optimizer and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        start_epoch = checkpoint.get('epoch', 0)

        best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        
        print(f"Loaded checkpoint from {filepath}. Resuming at epoch {start_epoch}.")
        print(f"Previous best_val_f1: {best_val_f1:.4f}. Epochs without improvement: {epochs_no_improve}.")
        return start_epoch, best_val_f1, epochs_no_improve
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting from scratch.")

        try:
            os.remove(filepath)
            print(f"Removed corrupted checkpoint: {filepath}")
        except OSError:
            pass 
        return 0, 0.0, 0 


def calculate_metrics(labels, preds, probs, class_names=['low', 'mid', 'high']):
    """
    Calculates all necessary classification metrics.
    """

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    weighted_precision = precision_score(labels, preds, average='weighted', zero_division=0)
    weighted_recall = recall_score(labels, preds, average='weighted', zero_division=0)

    roc_auc = 0.0
    try:
        if len(np.unique(labels)) == len(class_names):
            roc_auc = roc_auc_score(labels, probs, multi_class='ovr', average='weighted')
        else:
            roc_auc = 0.0 
    except ValueError as e:
        print(f"Warning: Could not calculate ROC AUC. Error: {e}")
        roc_auc = 0.0
    report = classification_report(
        labels, 
        preds, 
        target_names=class_names, 
        zero_division=0
    )
    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'roc_auc': roc_auc,
        'report': report
    }
    
    return metrics