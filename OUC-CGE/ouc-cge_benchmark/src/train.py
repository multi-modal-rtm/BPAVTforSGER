import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import warnings
import numpy as np
import shutil

from utils import load_config, save_checkpoint, load_checkpoint, calculate_metrics
from dataloaders import create_dataloaders
from models import get_model

warnings.filterwarnings("ignore", category=UserWarning)

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    batches_processed = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        if batch is None:
            continue
            
        video = batch['video'].to(device)
        audio = batch['audio'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()

        data_mode = model.config['data_mode']
        if data_mode == 'video_only':
            outputs = model(video)
        elif data_mode == 'audio_only':
            outputs = model(audio)
        else: 
            outputs = model(video, audio)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batches_processed += 1
        
    return total_loss / batches_processed if batches_processed > 0 else 0.0

def evaluate(model, dataloader, device):
    """Evaluates the model on a given dataset."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if batch is None:
                continue

            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)

            data_mode = model.config['data_mode']
            if data_mode == 'video_only':
                outputs = model(video)
            elif data_mode == 'audio_only':
                outputs = model(audio)
            else:
                outputs = model(video, audio)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def main(config_path):
    """Main training and evaluation loop."""
    config = load_config(config_path)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    shutil.copy(config_path, os.path.join(config['results_dir'], 'config.yaml'))
    print(f"Configuration file saved to {config['results_dir']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Running experiment for model: {config['model_name']}")

    train_loader, val_loader, _ = create_dataloaders(config)
    model = get_model(config).to(device)
    model.config = config

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=config['learning_rate'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    
    class_weights = torch.tensor(config['class_weights'], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    checkpoint_path = os.path.join(config['results_dir'], 'latest_model.pth')
    start_epoch = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device) 
    
    best_val_f1 = 0.0
    if start_epoch > 0 and os.path.exists(os.path.join(config['results_dir'], 'best_model.pth')): 
        best_checkpoint = torch.load(os.path.join(config['results_dir'], 'best_model.pth'), map_location=device)
        if 'best_val_f1' in best_checkpoint:
            best_val_f1 = best_checkpoint['best_val_f1']


    for epoch in range(start_epoch, config['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        labels, preds, probs = evaluate(model, val_loader, device)
        
        if len(labels) == 0:
            print("Validation failed: No valid samples were found.")
            continue

        metrics = calculate_metrics(labels, preds, probs, class_names=['low', 'mid', 'high'])
        val_f1 = metrics['macro_f1']

        state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'best_val_f1': best_val_f1}
        save_checkpoint(state, filename=checkpoint_path)

        if val_f1 > best_val_f1:
            print(f"Validation F1 improved ({best_val_f1:.4f} --> {val_f1:.4f}). Saving best model...")
            best_val_f1 = val_f1
            state['best_val_f1'] = best_val_f1 
            best_model_path = os.path.join(config['results_dir'], 'best_model.pth')
            save_checkpoint(state, filename=best_model_path)

        scheduler.step()

    print("\n--- Training finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and benchmark models on the OUC-CGE dataset.")
    parser.add_argument('--config', type=str, required=True, help="Path to the experiment's .yaml configuration file.")
    args = parser.parse_args()
    main(args.config)