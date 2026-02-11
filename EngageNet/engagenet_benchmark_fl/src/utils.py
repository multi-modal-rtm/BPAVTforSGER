import torch
import os
import logging
from omegaconf import DictConfig
import numpy as np
from sklearn.metrics import classification_report, f1_score
import torch.optim.lr_scheduler as lr_scheduler
from warmup_scheduler import GradualWarmupScheduler

logger = logging.getLogger(__name__)

def set_seed(seed):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    logger.info(f"Set random seed to {seed}")

def get_optimizer(model, config: DictConfig):
    """Creates the optimizer based on the config."""
    optimizer_name = config.optimizer.lower()
    if optimizer_name == "adamw":
        logger.info("Using AdamW optimizer.")
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_scheduler(optimizer, config: DictConfig, steps_per_epoch: int):
    """Creates the learning rate scheduler."""
    scheduler_name = config.lr_scheduler.lower()
    if scheduler_name == "cosineannealinglr":
        logger.info(f"Using CosineAnnealingLR with a {config.lr_warmup_epochs}-epoch warmup.")
        scheduler_cosine = lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=(config.epochs - config.lr_warmup_epochs) * steps_per_epoch,
            eta_min=config.min_lr
        )
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=config.lr_warmup_epochs * steps_per_epoch,
            after_scheduler=scheduler_cosine
        )
        return scheduler
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

def save_checkpoint(model, optimizer, epoch, val_loss, val_f1, config: DictConfig, is_best):
    """Saves the model checkpoint."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_f1': val_f1
    }
    
    output_dir = os.path.join(config.output_dir, config.model.name)
    os.makedirs(output_dir, exist_ok=True)
    
    latest_path = os.path.join(output_dir, 'latest_checkpoint.pth')
    torch.save(state, latest_path)

    if is_best:
        best_path = os.path.join(output_dir, 'best_checkpoint.pth')
        torch.save(state, best_path)

def load_checkpoint(model, optimizer, config, device):
    """
    Loads a checkpoint to resume training, correctly identifying the best score
    and falling back to the best checkpoint if the latest is missing.
    """
    latest_path = os.path.join(config.output_dir, config.model.name, 'latest_checkpoint.pth')
    best_path = os.path.join(config.output_dir, config.model.name, 'best_checkpoint.pth')
    
    start_epoch = 1
    best_val_f1 = 0.0

    if os.path.exists(best_path):
        try:
            best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            best_val_f1 = best_checkpoint.get('val_f1', 0.0) 
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
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resuming training from epoch {start_epoch}.")
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_to_load}: {e}. Starting from scratch.")
            start_epoch = 1
            best_val_f1 = 0.0 
    else:
        logger.info("No checkpoint found. Starting training from scratch.")

    return start_epoch, best_val_f1

def get_classification_report(labels, preds):
    """
    Calculates and formats a classification report and returns the macro F1-score.
    Returns exactly two values: the report string and the macro F1 float.
    """
    if len(np.unique(labels)) < 2 or len(np.unique(preds)) < 2:
        return "Classification report cannot be generated (single class in predictions).", 0.0

    class_names = ['Barely-engaged', 'Engaged', 'Highly-Engaged', 'Not-Engaged']
    
    labels = np.array(labels).astype(int)
    preds = np.array(preds).astype(int)

    unique_labels_in_data = np.unique(np.concatenate((labels, preds)))
    target_names = [class_names[i] for i in unique_labels_in_data if i < len(class_names)]

    report = classification_report(
        labels,
        preds,
        target_names=target_names,
        zero_division=0,
        digits=4
    )
    
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
    return report, macro_f1