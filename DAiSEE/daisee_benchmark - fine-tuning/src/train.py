"""
Main training script for the DAiSEE benchmark.
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import yaml
import argparse
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd 

from dataloaders import create_dataloaders
from models.models_attention import MultiTaskModel 
from utils import calculate_classification_metrics, FocalLoss

def train_one_epoch(model, dataloader, optimizer, criteria, device, config):
    """
    Runs a single epoch of training.
    """
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        if batch is None: continue 
        
        video, labels = batch

        if video.dim() != 5:
            print(f"Warning: Skipping malformed validation batch with shape {video.shape}")
            continue

        video, labels = video.to(device), labels.to(device)
      
        optimizer.zero_grad()

        (logits_b, logits_e, logits_c, logits_f) = model(video)

        labels = labels.long()
        loss_b = criteria[0](logits_b, labels[:, 0])
        loss_e = criteria[1](logits_e, labels[:, 1])
        loss_c = criteria[2](logits_c, labels[:, 2])
        loss_f = criteria[3](logits_f, labels[:, 3])

        loss = loss_b + loss_e + loss_c + loss_f
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criteria, device, config):
    """
    Runs evaluation on the validation set.
    """
    model.eval()
    total_loss = 0.0
    all_logits_b, all_logits_e, all_logits_c, all_logits_f = [], [], [], []
    all_labels_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if batch is None: continue 

            video, labels = batch

            if video.dim() != 5:
                print(f"Warning: Skipping malformed validation batch with shape {video.shape}")
                continue

            video, labels = video.to(device), labels.to(device)

            (logits_b, logits_e, logits_c, logits_f) = model(video)

            labels = labels.long()
            loss_b = criteria[0](logits_b, labels[:, 0])
            loss_e = criteria[1](logits_e, labels[:, 1])
            loss_c = criteria[2](logits_c, labels[:, 2])
            loss_f = criteria[3](logits_f, labels[:, 3])
            loss = loss_b + loss_e + loss_c + loss_f
            total_loss += loss.item()

            all_logits_b.append(logits_b.cpu())
            all_logits_e.append(logits_e.cpu())
            all_logits_c.append(logits_c.cpu())
            all_logits_f.append(logits_f.cpu())
            all_labels_list.append(labels.cpu())

    avg_loss = total_loss / len(dataloader)

    all_logits = [
        torch.cat(all_logits_b, dim=0).numpy(),
        torch.cat(all_logits_e, dim=0).numpy(),
        torch.cat(all_logits_c, dim=0).numpy(),
        torch.cat(all_logits_f, dim=0).numpy()
    ]
    all_labels = torch.cat(all_labels_list, dim=0).numpy()

    metrics = calculate_classification_metrics(all_logits, all_labels)

    metrics['val_loss'] = avg_loss
    return metrics

def main(config_path, resume_from):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders = create_dataloaders(config)

    model_name = config['model_name']

    num_frames = config['training'].get('num_frames', 32) 
    model = MultiTaskModel(
        model_name=model_name, 
        num_frames=num_frames, 
        output_attentions=False
    )
    
    model.to(device)


    base_lr = config['training'].get('learning_rate', 2e-5)
    weight_decay = config['training'].get('weight_decay', 1e-2)

    backbone_lr = base_lr / 10.0 

    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('base_model.'):
            backbone_params.append(param)
        else:
            classifier_params.append(param)

    optimizer_grouped_parameters = [
        {
            'params': backbone_params,
            'lr': backbone_lr,
            'weight_decay': weight_decay
        },
        {
            'params': classifier_params,
            'lr': base_lr,
            'weight_decay': 0.0
        }
    ]
    
    print(f"Optimizer Setup:")
    print(f"  - Backbone LR: {backbone_lr:.1e} (Weight Decay: {weight_decay})")
    print(f"  - Classifier LR: {base_lr:.1e} (Weight Decay: 0.0)")

    optimizer = AdamW(optimizer_grouped_parameters, lr=base_lr)

    warmup_epochs = config['training'].get('warmup_epochs', 5) 
    total_epochs = config['training']['epochs']

    if total_epochs <= warmup_epochs:
        print("Warning: total_epochs <= warmup_epochs. Using LinearLR for all epochs.")
        scheduler_warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=total_epochs)
        scheduler = scheduler_warmup
    else:
        scheduler_warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs)

        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=(total_epochs - warmup_epochs))

        scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])
    
    criteria = [FocalLoss(gamma=3).to(device) for _ in range(4)]

    log_path = output_dir / "training_log.csv"
    log_header = [
        "Epoch", "Train_Loss", "Val_Loss", "Val_F1_Avg", 
        "Val_F1_Boredom", "Val_F1_Engagement", "Val_F1_Confusion", "Val_F1_Frustration",
        "Val_Acc_Avg", "Val_Acc_Boredom", "Val_Acc_Engagement", "Val_Acc_Confusion", "Val_Acc_Frustration"
    ]

    start_epoch = 1
    best_metric = -1.0
    epochs_no_improve = 0
    patience = config['training'].get('patience', 10)     
    checkpoint_path = None
    if resume_from:
        if resume_from == 'best':
            checkpoint_path = output_dir / 'best_model.pth'
        elif resume_from == 'latest':
            checkpoint_path = output_dir / 'latest_model.pth'
        else:
            print(f"Warning: Invalid resume_from value '{resume_from}'. Must be 'best' or 'latest'. Starting from scratch.")

    if checkpoint_path and checkpoint_path.exists():
        print(f"--- Resuming training from {checkpoint_path} ---")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        ckpt_num_frames = checkpoint.get('num_frames', num_frames)
        if ckpt_num_frames != num_frames:
            print(f"CRITICAL ERROR: Checkpoint num_frames ({ckpt_num_frames}) does not match config num_frames ({num_frames}).")
            print("This will cause a crash. Please use a matching config or checkpoint.")
            return # Exit
            
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_metric = checkpoint.get('best_metric', -1.0)
            epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
            
            print(f"Successfully loaded checkpoint. Resuming from Epoch {start_epoch}.")
            print(f"Best metric from previous run: {best_metric:.4f}")
            
        except Exception as e:
            print(f"Error loading checkpoint state: {e}. Starting from scratch.")
            start_epoch = 1
            best_metric = -1.0
            epochs_no_improve = 0
            
    else:
        if resume_from:
            print(f"Warning: Checkpoint '{resume_from}' not found at {checkpoint_path}. Starting from scratch.")
        if not log_path.exists():
            with open(log_path, 'w') as f:
                f.write(','.join(log_header) + '\n')

    print(f"--- Starting training for {model_name} (Frames: {num_frames}) ---")

    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        print(f"\n--- Epoch {epoch}/{config['training']['epochs']} ---")
        
        train_loss = train_one_epoch(model, dataloaders['Train'], optimizer, criteria, device, config)
        val_metrics = evaluate(model, dataloaders['Validation'], criteria, device, config)
        
        scheduler.step()

        current_metric = val_metrics['f1_macro_avg']
        
        print(f"Epoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
        print(f"  Val F1 Avg: {current_metric:.4f} (Best: {best_metric:.4f})")
        print(f"  Current LR: {optimizer.param_groups[-1]['lr']:.2e}") 
        
        log_data = [
            epoch, train_loss, val_metrics['val_loss'], current_metric,
            val_metrics['f1_macro_boredom'], val_metrics['f1_macro_engagement'],
            val_metrics['f1_macro_confusion'], val_metrics['f1_macro_frustration'],
            val_metrics['accuracy_avg'], val_metrics['accuracy_boredom'],
            val_metrics['accuracy_engagement'], val_metrics['accuracy_confusion'],
            val_metrics['accuracy_frustration']
        ]
        with open(log_path, 'a') as f:
            f.write(','.join(map(str, log_data)) + '\n')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_metric': best_metric,
            'epochs_no_improve': epochs_no_improve,
            'num_frames': num_frames
        }, output_dir / 'latest_model.pth')

        if current_metric > best_metric:
            print(f"  New best model found! Saving to {output_dir / 'best_model.pth'}")
            best_metric = current_metric
            epochs_no_improve = 0

            shutil.copyfile(output_dir / 'latest_model.pth', output_dir / 'best_model.pth')
            
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break
            
    print(f"--- Training complete. Best F1 Avg: {best_metric:.4f} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the DAiSEE benchmark.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the .yaml configuration file.")
    parser.add_argument("--resume_from", type=str, default=None,
                        choices=['best', 'latest'],
                        help="Resume training from 'best' or 'latest' checkpoint in the config's output_dir.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
    else:
        main(args.config, args.resume_from)