import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import random
import numpy as np
import os
import logging
from omegaconf import DictConfig
from sklearn.metrics import classification_report, f1_score
import warnings
import _pickle 

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_optimizer(model: nn.Module, config: DictConfig) -> torch.optim.Optimizer:
    """Creates an AdamW optimizer with weight decay."""
    if config.optimizer == "AdamW":
        return AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")

def get_scheduler(optimizer: torch.optim.Optimizer, config: DictConfig, steps_per_epoch: int = None):
    """
    Creates a learning rate scheduler with a linear warmup phase
    followed by a cosine annealing phase.
    
    If steps_per_epoch is None, creates an EPOCH-based scheduler.
    If steps_per_epoch is provided, creates a STEP-based scheduler.
    """
    
    warmup_epochs = config.lr_warmup_epochs
    total_epochs = config.epochs
    main_epochs = total_epochs - warmup_epochs

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-6, 
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=main_epochs,
        eta_min=config.min_lr 
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs] 
    )
    logger.info(f"Using Epoch-based CosineAnnealingLR with a {warmup_epochs}-epoch warmup.")
    return scheduler


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_val_f1, config, is_best):
    """Saves the model checkpoint, including the scheduler state."""
    model_output_dir = os.path.join(config.output_dir, config.model.name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    state = {
        'epoch': epoch, 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'best_val_f1': best_val_f1, 
    }

    latest_path = os.path.join(model_output_dir, 'latest_checkpoint.pth')
    torch.save(state, latest_path)

    if is_best:
        best_path = os.path.join(model_output_dir, 'best_checkpoint.pth')
        torch.save(state, best_path)

def load_checkpoint(model, optimizer, scheduler, config, device):
    """
    Loads a checkpoint to resume training, correctly loading all states
    and finding the best F1 score.
    """
    model_output_dir = os.path.join(config.output_dir, config.model.name)
    latest_path = os.path.join(model_output_dir, 'latest_checkpoint.pth')
    best_path = os.path.join(model_output_dir, 'best_checkpoint.pth')
    
    start_epoch = 1
    best_val_f1 = 0.0

    if os.path.exists(best_path):
        try:
            best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            best_val_f1 = best_checkpoint.get('best_val_f1', 0.0) 
            
            logger.info(f"Found best validation F1 from previous run: {best_val_f1:.4f}")
        except Exception as e:
            logger.warning(f"Could not load best checkpoint: {e}. Starting with best F1 of 0.0")

    checkpoint_to_load = None
    if os.path.exists(latest_path):
        checkpoint_to_load = latest_path
        logger.info(f"Found latest checkpoint. Attempting to resume from: {latest_path}")
    elif os.path.exists(best_path):
        checkpoint_to_load = best_path
        logger.info(f"Latest checkpoint not found. Attempting to resume from best checkpoint: {best_path}")

    if checkpoint_to_load:
        try:
            checkpoint = torch.load(checkpoint_to_load, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("Resumed scheduler state.")
            else:
                logger.warning("Scheduler state not found in checkpoint. Scheduler will start from scratch.")
                
            start_epoch = checkpoint.get('epoch', 1)

            if best_val_f1 == 0.0:
                 best_val_f1 = checkpoint.get('best_val_f1', 0.0)
 
            if checkpoint_to_load == latest_path:
                start_epoch += 1 
                
            logger.info(f"Resuming training from epoch {start_epoch}.")
            
        except (RuntimeError, KeyError, TypeError) as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_to_load}: {e}. Starting from scratch.")
            start_epoch = 1
            best_val_f1 = 0.0 
        except _pickle.UnpicklingError:
             logger.warning(f"Checkpoint file at {checkpoint_to_load} is corrupt. Starting from scratch.")
             start_epoch = 1
             best_val_f1 = 0.0
    else:
        logger.info("No checkpoint found. Starting training from scratch.")

    return start_epoch, best_val_f1

def get_classification_report(all_labels, all_preds):
    """
    Generates a classification report and calculates macro F1-score.
    Handles warnings for classes with no predictions.
    """
    target_names = ['Barely-engaged', 'Engaged', 'Highly-Engaged', 'Not-Engaged']
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        
        report = classification_report(
            all_labels, 
            all_preds, 
            target_names=target_names,
            zero_division=0
        )
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    if 'nan' in report:
        report += "\nWarning: 'nan' values present in report. Some classes may have 0 support."

    return report, macro_f1