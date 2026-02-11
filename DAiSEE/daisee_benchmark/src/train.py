import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import yaml
import argparse
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np

from dataloaders import create_dataloaders
from models import (
    get_vit_model, get_videomae_model, get_swin_model, 
    get_mvit_model, get_maxvit_model, get_timesformer_model
)
from utils import calculate_classification_metrics

def get_model(model_name, num_outputs=16):
    """Factory function to get the correct model for the benchmark."""
    if model_name == "vit":
        return get_vit_model(num_outputs)
    elif model_name == "videomae":
        return get_videomae_model(num_outputs)
    elif model_name == "swin":
        return get_swin_model(num_outputs)
    elif model_name == "mvit":
        return get_mvit_model(num_outputs)
    elif model_name == "maxvit":
        return get_maxvit_model(num_outputs)
    elif model_name == "timesformer":
        return get_timesformer_model(num_outputs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_one_epoch(model, dataloader, optimizer, criteria, device, config):
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        if batch is None or len(batch[0]) == 0: continue
        
        video, labels = batch
        video, labels = video.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if config['model_name'] in ['vit', 'swin', 'mvit', 'maxvit']:
            b_frames, t, c, h, w = video.shape
            video_reshaped = video.view(b_frames * t, c, h, w)
            frame_logits = model(video_reshaped)
            logits = frame_logits.view(b_frames, t, -1).mean(dim=1)

        elif config['model_name'] in ['videomae', 'timesformer']:
            logits = model(pixel_values=video)

        elif config['model_name'] == 'mbt':
            logits = model(video)
            
        else:
            raise NotImplementedError(f"Forward pass for model {config['model_name']} not implemented.")

        b = labels.shape[0] 
        logits = logits.view(b, 4, 4) 
        labels = labels.long()
        loss = sum(criteria[i](logits[:, i, :], labels[:, i]) for i in range(4))
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criteria, device, config):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if batch is None or len(batch[0]) == 0: continue

            video, labels = batch
            video, labels = video.to(device), labels.to(device)

            if config['model_name'] in ['vit', 'swin', 'mvit', 'maxvit']:
                b_frames, t, c, h, w = video.shape
                video_reshaped = video.view(b_frames * t, c, h, w)
                frame_logits = model(video_reshaped)
                logits = frame_logits.view(b_frames, t, -1).mean(dim=1)

            elif config['model_name'] in ['videomae', 'timesformer']:
                logits = model(pixel_values=video)
                
            else:
                raise NotImplementedError(f"Forward pass for model {config['model_name']} not implemented.")
            
            b = labels.shape[0]
            logits = logits.view(b, 4, 4)
            labels = labels.long()
            loss = sum(criteria[i](logits[:, i, :], labels[:, i]) for i in range(4))
            total_loss += loss.item()
            
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    metrics = calculate_classification_metrics(all_logits, all_labels)
    metrics['loss'] = avg_loss
    return metrics

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    shutil.copy(config_path, output_dir / 'config.yaml')
    print(f"Saved configuration file to {output_dir}")

    dataloaders = create_dataloaders(config)
    if 'Train' not in dataloaders:
        print("Training dataloader not found. Exiting.")
        return

    model = get_model(config['model_name'], num_outputs=16).to(device)

    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    warmup_epochs = 5
    total_epochs = int(config['training']['epochs'])
    
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    boredom_weights = torch.tensor([0.049, 0.071, 0.112, 0.768], device=device, dtype=torch.float32)
    engagement_weights = torch.tensor([0.843, 0.135, 0.011, 0.011], device=device, dtype=torch.float32)
    confusion_weights = torch.tensor([0.015, 0.043, 0.125, 0.817], device=device, dtype=torch.float32)
    frustration_weights = torch.tensor([0.008, 0.036, 0.176, 0.780], device=device, dtype=torch.float32)
    
    criteria = [
        nn.CrossEntropyLoss(weight=boredom_weights),
        nn.CrossEntropyLoss(weight=engagement_weights),
        nn.CrossEntropyLoss(weight=confusion_weights),
        nn.CrossEntropyLoss(weight=frustration_weights)
    ]

    start_epoch = 0
    best_metric = 0.0
    
    patience = 10
    epochs_no_improve = 0
    
    checkpoint_path = output_dir / 'latest_model.pth'

    if checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_metric = checkpoint.get('best_metric', 0)
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Resumed scheduler state.")
        print(f"Resumed from epoch {start_epoch}. Best exact match accuracy so far: {best_metric:.4f}")
    else:
        print("Starting training from scratch.")

    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['training']['epochs']} ---")
        
        train_loss = train_one_epoch(model, dataloaders['Train'], optimizer, criteria, device, config)
        print(f"Train Loss: {train_loss:.4f}")
        
        scheduler.step()
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.8f}")
        
        if 'Validation' in dataloaders:
            val_metrics = evaluate(model, dataloaders['Validation'], criteria, device, config)
            current_metric = val_metrics['exact_match_accuracy']
            print(f"Validation Exact Match Accuracy: {current_metric:.4f}")
        else:
            current_metric = 0 
            torch.save({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'best_metric': best_metric, 'epochs_no_improve': epochs_no_improve
            }, output_dir / 'latest_model.pth')
            continue

        if current_metric > best_metric:
            best_metric = current_metric
            epochs_no_improve = 0
            print(f"New best model saved with Exact Match Accuracy: {best_metric:.4f}")
            torch.save({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'best_metric': best_metric, 'epochs_no_improve': epochs_no_improve
            }, output_dir / 'best_model.pth')
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation accuracy for {epochs_no_improve} epoch(s).")
        
        torch.save({
            'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
            'best_metric': best_metric, 'epochs_no_improve': epochs_no_improve
        }, output_dir / 'latest_model.pth')

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

    print("\n--- Final Evaluation on Test Set using Best Model ---")
    if (output_dir / 'best_model.pth').exists():
        best_model_checkpoint = torch.load(output_dir / 'best_model.pth')
        model.load_state_dict(best_model_checkpoint['model_state_dict'])
        print("Loaded best model for final evaluation.")
    else:
        print("No best model found. Using last trained model for evaluation.")

    if 'Test' in dataloaders:
        test_metrics = evaluate(model, dataloaders['Test'], criteria, device, config)
        with open(output_dir / 'test_results.json', 'w') as f:
            json.dump(test_metrics, f, indent=4)
        
        print("\nTest Metrics (from best model):")
        for key, val in test_metrics.items():
            if isinstance(val, (float, np.floating)):
                print(f"  {key}: {val:.4f}")
    
    print(f"\nBenchmarking for '{config['model_name']}' complete. Results saved in {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and benchmark a model on the DAiSEE dataset.")
    parser.add_argument('--config', type=str, required=True, help="Path to the model's YAML config file.")
    args = parser.parse_args()
    main(args.config)