import torch
import random
import numpy as np
import os
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

class WarmupLR(_LRScheduler):
    """
    A learning rate scheduler with a linear warmup phase.
    """
    def __init__(self, optimizer, total_iters, warmup_iters, last_epoch=-1):
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            return [base_lr * self.last_epoch / self.warmup_iters for base_lr in self.base_lrs]
        else:
            return [base_lr for base_lr in self.base_lrs]

def set_seed(seed):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")

def save_checkpoint(model, optimizer, epoch, val_loss, is_best, config):
    """
    Saves the model checkpoint.
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': config
    }
    
    output_dir = os.path.join(config.output_dir, config.model.name)
    os.makedirs(output_dir, exist_ok=True)
    
    latest_path = os.path.join(output_dir, 'latest_checkpoint.pth')
    torch.save(state, latest_path)

    if is_best:
        best_path = os.path.join(output_dir, 'best_checkpoint.pth')
        torch.save(state, best_path)
        logger.info(f"Saved best checkpoint to {best_path} (val_loss: {val_loss:.4f})")

def load_checkpoint(model, optimizer, config, device):
    """
    Loads a checkpoint to resume training, correctly identifying the best score
    and falling back to the best checkpoint if the latest is missing.
    """
    latest_path = os.path.join(config.output_dir, config.model.name, 'latest_checkpoint.pth')
    best_path = os.path.join(config.output_dir, config.model.name, 'best_checkpoint.pth')
    
    start_epoch = 1
    best_val_loss = float('inf')
    
    resume_path = None
    if os.path.exists(latest_path):
        resume_path = latest_path
        logger.info(f"Found latest checkpoint. Attempting to resume from: {resume_path}")
    elif os.path.exists(best_path):
        resume_path = best_path
        logger.warning(f"Latest checkpoint not found. Falling back to best checkpoint to resume: {resume_path}")
    
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Successfully resumed model and optimizer state from epoch {checkpoint['epoch']}.")

    if os.path.exists(best_path):
        best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        best_val_loss = best_checkpoint['val_loss']
        logger.info(f"Best validation loss from previous runs: {best_val_loss:.4f}")
    
    if not resume_path:
        logger.info("No checkpoint found. Starting training from scratch.")

    return start_epoch, best_val_loss


def get_classification_report(y_true, y_pred):
    """
    Generates a classification report from sklearn.
    """
    target_names = ['Barely-engaged', 'Engaged', 'Highly-Engaged', 'Not-Engaged']
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    return report


def get_scheduler(optimizer, config, steps_per_epoch):
    """
    Creates a learning rate scheduler with an optional warmup phase.
    """
    main_scheduler_name = config.lr_scheduler
    total_epochs = config.epochs
    
    if main_scheduler_name == "CosineAnnealingLR":
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=(total_epochs - config.lr_warmup_epochs) * steps_per_epoch,
            eta_min=config.min_lr
        )
    elif main_scheduler_name == "StepLR":
        main_scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        raise NotImplementedError(f"Scheduler {main_scheduler_name} not implemented.")

    if config.lr_warmup_epochs > 0:
        warmup_scheduler = WarmupLR(
            optimizer,
            total_iters=config.lr_warmup_epochs * steps_per_epoch,
            warmup_iters=config.lr_warmup_epochs * steps_per_epoch
        )
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup_scheduler, main_scheduler])
        logger.info(f"Using {main_scheduler_name} with a {config.lr_warmup_epochs}-epoch warmup.")
    else:
        scheduler = main_scheduler
        logger.info(f"Using {main_scheduler_name} scheduler without warmup.")
        
    return scheduler

def get_optimizer(model, config):
    """
    Creates an optimizer for the given model.
    """
    if config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {config.optimizer} not implemented.")
        
    logger.info(f"Using {config.optimizer} optimizer.")
    return optimizer

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''Calculates the mixed loss'''
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)