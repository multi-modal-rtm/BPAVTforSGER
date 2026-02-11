import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import warnings
import numpy as np
import pandas as pd
import shutil
from sklearn.metrics import accuracy_score

from utils import load_config, save_checkpoint, load_checkpoint, calculate_metrics
from dataloaders import create_dataloaders
from models import get_model

warnings.filterwarnings("ignore", category=UserWarning)

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in progress_bar:
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

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix(loss=loss.item())

    if not all_labels:
        print("Warning: No labels processed in train_one_epoch. Returning 0 accuracy.")
        return 0.0, 0.0

    avg_acc = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)

    return avg_loss, avg_acc

def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on a given dataset."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    total_val_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
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
            
            if len(dataloader) == 0:
                print("Warning: Empty evaluation dataloader.")
                return 0.0, np.array([]), np.array([]), np.array([])
                
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    avg_val_loss = total_val_loss / len(dataloader)
    
    return avg_val_loss, np.array(all_labels), np.array(all_preds), np.array(all_probs)

def main(config_path):
    """Main training and evaluation loop."""
    config = load_config(config_path)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    shutil.copy(config_path, os.path.join(config['results_dir'], 'config.yaml'))
    print(f"Configuration file saved to {config['results_dir']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Running experiment for model: {config['model_name']}")

    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    if not train_loader or not val_loader:
        print("Error: Training or validation dataloader is empty. Check data paths and CSV files.")
        return

    model = get_model(config).to(device)
    model.config = config 

    backbone_params = [p for name, p in model.named_parameters() if 'backbone' in name and p.requires_grad]
    glance_focus_params = [p for name, p in model.named_parameters() if ('glance_net' in name or 'focus_net' in name) and p.requires_grad]
    head_params = [p for name, p in model.named_parameters() if 'backbone' not in name and 'glance_net' not in name and 'focus_net' not in name and p.requires_grad]

    param_groups_config = [
        {'params': backbone_params, 'lr': config['learning_rate']},
        {'params': glance_focus_params, 'lr': config['learning_rate']},
        {'params': head_params, 'lr': config['learning_rate'] * 10}
    ]
    param_groups = [group for group in param_groups_config if group['params']]
    
    if not param_groups:
        print("Warning: No trainable parameters found. Check model's requires_grad settings.")
        param_groups = model.parameters()
        
    optimizer = torch.optim.AdamW(param_groups)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    
    class_weights = torch.tensor(config['class_weights'], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    checkpoint_path = os.path.join(config['results_dir'], 'latest_model.pth')
    
    start_epoch, best_val_f1, epochs_no_improve = load_checkpoint(
        checkpoint_path, model, optimizer, scheduler, device
    ) 
    

    log_data = []
    log_file_path = os.path.join(config['results_dir'], "training_log.csv")

    if start_epoch > 0 and os.path.exists(log_file_path):
        print(f"Resuming training. Loading previous log data from {log_file_path}")
        try:
            log_df_old = pd.read_csv(log_file_path)

            log_data = log_df_old.to_dict('records')
            print(f"Loaded {len(log_data)} previous log entries.")
        except Exception as e:
            print(f"Warning: Could not load previous log file. Starting a new log. Error: {e}")
            log_data = []

    patience = config.get('patience', 10) 

    for epoch in range(start_epoch, config['epochs']):
        print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, labels, preds, probs = evaluate(model, val_loader, criterion, device)
        
        if len(labels) == 0:
            print("Validation failed: No valid samples were found.")
            continue

        metrics = calculate_metrics(labels, preds, probs, class_names=['low', 'mid', 'high'])
        
        val_f1 = metrics['weighted_f1'] 
        val_acc = metrics['accuracy']
        val_precision = metrics['weighted_precision']
        val_recall = metrics['weighted_recall']
        val_roc_auc = metrics['roc_auc']

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1: {val_f1:.4f}")
        print(f"Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val ROC AUC: {val_roc_auc:.4f}")

        log_epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_roc_auc': val_roc_auc
        }
        log_data.append(log_epoch_data)

        state = {
            'epoch': epoch + 1, 
            'state_dict': model.state_dict(), 
            'optimizer': optimizer.state_dict(), 
            'scheduler': scheduler.state_dict(), 
            'best_val_f1': best_val_f1,
            'epochs_no_improve': epochs_no_improve
        }
        save_checkpoint(state, filename=checkpoint_path)

        if val_f1 > best_val_f1:
            print(f"Validation F1 improved ({best_val_f1:.4f} --> {val_f1:.4f}). Saving best model...")
            best_val_f1 = val_f1
            epochs_no_improve = 0 

            state['best_val_f1'] = best_val_f1 
            state['epochs_no_improve'] = epochs_no_improve 
            
            best_model_path = os.path.join(config['results_dir'], 'best_model.pth')
            save_checkpoint(state, filename=best_model_path)
        else:
            epochs_no_improve += 1
            print(f"Validation F1 did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"\n--- Early stopping triggered after {patience} epochs with no improvement ---")
            break

        scheduler.step()

        log_df = pd.DataFrame(log_data)
        try:
            log_df.to_csv(log_file_path, index=False)

            if (epoch + 1) % 10 == 0: 
                 print(f"Training log (re)saved to {log_file_path}")
        except Exception as e:
            print(f"Failed to save training log: {e}")


    print("\n--- Training finished ---")
    
    log_df = pd.DataFrame(log_data)
    log_file_path = os.path.join(config['results_dir'], "training_log.csv")
    try:
        log_df.to_csv(log_file_path, index=False)
        print(f"Final training log saved to {log_file_path}")
    except Exception as e:
        print(f"Failed to save final training log: {e}")

    print("\n--- Evaluating on Test Set with Best Model ---")
    best_model_path = os.path.join(config['results_dir'], 'best_model.pth')
    if os.path.exists(best_model_path):
        best_checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(best_checkpoint['state_dict'])
        
        if not test_loader:
             print("Test dataloader is empty. Skipping test set evaluation.")
        else:
            test_loss, test_labels, test_preds, test_probs = evaluate(model, test_loader, criterion, device)
            
            if len(test_labels) > 0:
                test_metrics = calculate_metrics(test_labels, test_preds, test_probs, class_names=['low', 'mid', 'high'])
                
                print("--- Test Set Results ---")
                print(f"Test Loss: {test_loss:.4f}")
                print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
                print(f"Test Weighted F1: {test_metrics['weighted_f1']:.4f}")
                print(f"Test Precision: {test_metrics['weighted_precision']:.4f}")
                print(f"Test Recall: {test_metrics['weighted_recall']:.4f}")
                print(f"Test ROC AUC: {test_metrics['roc_auc']:.4f}")
                
                test_metrics_path = os.path.join(config['results_dir'], 'test_results.txt')
                with open(test_metrics_path, 'w') as f:
                    f.write("--- Test Set Results ---\n")
                    f.write(f"Test Loss: {test_loss:.4f}\n")
                    f.write(f"Test Accuracy: {test_metrics['accuracy']:.4f}\n")
                    f.write(f"Test Weighted F1: {test_metrics['weighted_f1']:.4f}\n")
                    f.write(f"Test Precision: {test_metrics['weighted_precision']:.4f}\n")
                    f.write(f"Test Recall: {test_metrics['weighted_recall']:.4f}\n")
                    f.write(f"Test ROC AUC: {test_metrics['roc_auc']:.4f}\n\n")
                    f.write("--- Classification Report ---\n")
                    f.write(test_metrics['report'])
                print(f"Test results saved to {test_metrics_path}")
                
            else:
                print("Test evaluation failed: No valid samples found.")
    else:
        print("No best model found. Skipping test set evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and benchmark models on the OUC-CGE dataset.")
    parser.add_argument('--config', type=str, required=True, help="Path to the experiment's .yaml configuration file.")
    args = parser.parse_args()
    main(args.config)