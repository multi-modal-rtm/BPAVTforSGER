import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import logging
import os
from tqdm import tqdm
import numpy as np

from .dataloaders import create_dataloader
from .models import create_model
from .losses import get_criterion
from .utils import (
    set_seed, get_optimizer, get_scheduler,
    save_checkpoint, load_checkpoint, get_classification_report
)

logger = logging.getLogger(__name__)

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for i, batch in enumerate(progress_bar):
        if batch is None:
            logger.warning("Skipping a batch because all samples failed to load.")
            continue
            
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        modality = dataloader.dataset.modality
        if modality == 'audiovisual':
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            outputs = model((video, audio))
        else:
            video = batch['video'].to(device)
            outputs = model(video)

        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        progress_bar.set_postfix(
            loss=total_loss / (i + 1),
            acc=total_correct / total_samples,
            lr=scheduler.get_last_lr()[0]
        )
        
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0
    return avg_loss, avg_acc

def validate(model, dataloader, criterion, device):
    """Evaluates the model on the validation set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            if batch is None:
                continue

            labels = batch['label'].to(device)
            
            modality = dataloader.dataset.modality
            if modality == 'audiovisual':
                video = batch['video'].to(device)
                audio = batch['audio'].to(device)
                outputs = model((video, audio))
            else:
                video = batch['video'].to(device)
                outputs = model(video)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if not all_labels:
        return 0.0, 0.0, "No valid samples found in validation set.", 0.0

    avg_loss = total_loss / len(dataloader)
    avg_acc = np.mean(np.array(all_preds) == np.array(all_labels))

    report, macro_f1 = get_classification_report(all_labels, all_preds)
    
    return avg_loss, avg_acc, report, macro_f1

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(config: DictConfig):
    """Main training and evaluation pipeline."""
    logger.info("Starting training pipeline...")
    
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_loader = create_dataloader(config, 'train')
    val_loader = create_dataloader(config, 'val')

    model = create_model(config).to(device)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, len(train_loader))
    criterion = get_criterion(config, device)
    

    start_epoch, best_val_f1 = load_checkpoint(model, optimizer, config, device)

    epochs_since_improvement = 0
    for epoch in range(start_epoch, config.epochs + 1):
        logger.info(f"--- Epoch {epoch}/{config.epochs} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Unpack the new return values from validate
        val_loss, val_acc, report, val_f1 = validate(model, val_loader, criterion, device)
        logger.info(f"Epoch {epoch} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro-F1: {val_f1:.4f}")
        logger.info(f"Validation Classification Report:\n{report}")
        

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            epochs_since_improvement = 0
            logger.info(f"New best validation F1: {best_val_f1:.4f}. Saving best model.")
            save_checkpoint(model, optimizer, epoch, val_loss, val_f1, config, is_best=True)
        else:
            epochs_since_improvement += 1

        save_checkpoint(model, optimizer, epoch, val_loss, val_f1, config, is_best=False)
            
        if epochs_since_improvement >= config.early_stopping_patience:
            logger.info(f"No improvement in validation F1 for {config.early_stopping_patience} epochs. Stopping training.")
            break

    logger.info("--- Training finished ---")

if __name__ == "__main__":
    main()

