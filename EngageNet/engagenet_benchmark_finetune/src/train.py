import os
import torch
import torch.nn as nn
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import yaml
import csv 
from torch.amp import GradScaler, autocast 

from .dataloaders import create_dataloader
from .models import create_model
from .utils import set_seed, get_optimizer, get_scheduler, save_checkpoint, load_checkpoint, get_classification_report
from .losses import get_criterion


logger = logging.getLogger(__name__)

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device, scaler, use_amp):
    """
    Trains the model for one epoch.
    Conditionally uses Automatic Mixed Precision if use_amp is True.
    """
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        video = batch['video'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=use_amp):
            if isinstance(batch.get('audio'), torch.Tensor):
                audio = batch['audio'].to(device)
                outputs = model((video, audio))
            else:
                outputs = model(video)

        loss = criterion(outputs.float(), labels)

        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
    
        scaler.update()
                
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
        progress_bar.set_postfix(
            loss=total_loss / (progress_bar.n + 1),
            acc=correct_predictions / total_samples,
            lr=optimizer.param_groups[0]['lr']
        )

    scheduler.step()
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = correct_predictions / total_samples
    return avg_loss, avg_acc

def validate(model, dataloader, criterion, device, use_amp):
    """
    Validates the model on the validation set.
    Conditionally uses Automatic Mixed Precision if use_amp is True.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            video = batch['video'].to(device)
            labels = batch['label'].to(device)
            
            with autocast(device_type=device.type, enabled=use_amp):
                if isinstance(batch.get('audio'), torch.Tensor):
                    audio = batch['audio'].to(device)
                    outputs = model((video, audio))
                else:
                    outputs = model(video)

            loss = criterion(outputs.float(), labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    report, macro_f1 = get_classification_report(all_labels, all_preds)
    val_acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    
    return avg_loss, val_acc, report, macro_f1

def setup_log_file(log_path):
    """Initializes the CSV log file with a header if it doesn't exist."""
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_macro_f1'])

def append_log(log_path, epoch, train_loss, train_acc, val_loss, val_acc, val_f1):
    """Appends a new row of metrics to the CSV log file."""
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, val_f1])

@hydra.main(config_path="../configs", config_name="base_finetune", version_base=None)
def main(config: DictConfig):
    """
    The main training pipeline.
    """
    logger.info("Starting training pipeline...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    train_loader = create_dataloader(config, 'train')
    val_loader = create_dataloader(config, 'val')
    
    model = create_model(config).to(device)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, steps_per_epoch=len(train_loader))
    criterion = get_criterion(config, device)
    
    start_epoch, best_val_f1 = load_checkpoint(model, optimizer, scheduler, config, device)

    use_amp = (config.model.name == 'MaxViT')
    logger.info(f"Using Automatic Mixed Precision (AMP) for {config.model.name}: {use_amp}")

    scaler = GradScaler(enabled=use_amp)

    model_output_dir = os.path.join(config.output_dir, config.model.name)
    os.makedirs(model_output_dir, exist_ok=True)
    log_file_path = os.path.join(model_output_dir, "training_log.csv")
    if start_epoch == 1:
        setup_log_file(log_file_path) 

    epochs_since_improvement = 0


    for epoch in range(start_epoch, config.epochs + 1):
        logger.info(f"--- Epoch {epoch}/{config.epochs} ---")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, scaler, use_amp)
        logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        val_loss, val_acc, report, val_f1 = validate(model, val_loader, criterion, device, use_amp)
        logger.info(f"Epoch {epoch} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro-F1: {val_f1:.4f}")
        logger.info(f"Validation Classification Report:\n{report}")

        append_log(log_file_path, epoch, train_loss, train_acc, val_loss, val_acc, val_f1)

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            logger.info(f"New best validation F1: {best_val_f1:.4f}. Saving best model.")
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_val_f1, config, is_best)

        if epochs_since_improvement >= config.early_stopping_patience:
            logger.info(f"No improvement in validation F1 for {config.early_stopping_patience} epochs. Stopping training.")
            break

if __name__ == "__main__":
    main()