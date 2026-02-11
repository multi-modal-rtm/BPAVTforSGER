import torch
import torch.nn as nn
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import numpy as np

from .dataloaders import create_dataloader
from .models import create_model
from .utils import (
    set_seed, get_optimizer, get_scheduler, save_checkpoint, 
    load_checkpoint, get_classification_report, mixup_data, mixup_criterion
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """
    Trains the model for one epoch using Mixup.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for i, batch in enumerate(progress_bar):
        if batch is None:
            logger.warning("Skipping a batch because all samples failed to load.")
            continue

        optimizer.zero_grad()
        
        video = batch['video'].to(device)
        labels = batch['label'].to(device)

        mixed_video, labels_a, labels_b, lam = mixup_data(video, labels, alpha=0.4, device=device)
        
        outputs = model(mixed_video)

        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_correct += (lam * (predicted == labels_a).sum().item() + (1 - lam) * (predicted == labels_b).sum().item())
        total_samples += labels.size(0)

        if total_samples > 0:
            progress_bar.set_postfix(
                loss=total_loss / (i + 1),
                acc=total_correct / total_samples,
                lr=scheduler.get_last_lr()[0]
            )
    
    return total_loss / len(dataloader), total_correct / total_samples if total_samples > 0 else 0

def validate(model, dataloader, criterion, device):
    """
    Validates the model (without Mixup).
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            if batch is None: continue

            video = batch['video'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(video)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    if not all_labels:
        return 0.0, 0.0, "Validation set was empty or all samples failed to load."

    val_loss = total_loss / len(dataloader)
    val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    report = get_classification_report(all_labels, all_preds)
    
    return val_loss, val_acc, report


@hydra.main(version_base=None, config_path="../configs", config_name="vit")
def main(config: DictConfig):
    """
    The main function to run the training pipeline.
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
    scheduler = get_scheduler(optimizer, config, len(train_loader))
    
    if 'class_weights' in config and config.class_weights is not None:
        class_weights = torch.tensor(config.class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"Applying class weights to loss function: {class_weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Using unweighted CrossEntropyLoss.")

    start_epoch, best_val_loss = load_checkpoint(model, optimizer, config, device)
    
    epochs_no_improve = 0
    patience = config.get('early_stopping_patience', 10)
    logger.info(f"Using early stopping with patience of {patience} epochs.")

    for epoch in range(start_epoch, config.epochs + 1):
        logger.info(f"--- Epoch {epoch}/{config.epochs} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        val_loss, val_acc, report = validate(model, val_loader, criterion, device)
        logger.info(f"Epoch {epoch} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        logger.info(f"Validation Classification Report:\n{report}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, val_loss=val_loss, is_best=True, config=config)
        else:
            epochs_no_improve += 1
            save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, val_loss=val_loss, is_best=False, config=config)

        if epochs_no_improve >= patience:
            logger.info(f"Validation loss has not improved for {patience} epochs. Stopping training early.")
            break

    logger.info("Training finished!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()











# import torch
# import torch.nn as nn
# from tqdm import tqdm
# import hydra
# from omegaconf import DictConfig, OmegaConf
# import logging
# import os
# import numpy as np

# from .dataloaders import create_dataloader
# from .models import create_model
# from .utils import (
#     set_seed, get_optimizer, get_scheduler, save_checkpoint, 
#     load_checkpoint, get_classification_report, mixup_data, mixup_criterion
# )

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
#     """
#     Trains the model for one epoch using Mixup.
#     """
#     model.train()
#     total_loss = 0.0
#     total_correct = 0
#     total_samples = 0
    
#     progress_bar = tqdm(dataloader, desc="Training")
#     for i, batch in enumerate(progress_bar):
#         if batch is None:
#             logger.warning("Skipping a batch because all samples failed to load.")
#             continue

#         optimizer.zero_grad()
        
#         video = batch['video'].to(device)
#         labels = batch['label'].to(device)
        
#         # --- ADDED: Apply Mixup ---
#         mixed_video, labels_a, labels_b, lam = mixup_data(video, labels, alpha=0.4, device=device)
        
#         outputs = model(mixed_video)
        
#         # --- ADDED: Calculate loss using Mixup criterion ---
#         loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#         scheduler.step()

#         total_loss += loss.item()
        
#         # --- Accuracy is calculated on the original, unmixed labels for clarity ---
#         _, predicted = torch.max(outputs.data, 1)
#         total_correct += (lam * (predicted == labels_a).sum().item() + (1 - lam) * (predicted == labels_b).sum().item())
#         total_samples += labels.size(0)

#         if total_samples > 0:
#             progress_bar.set_postfix(
#                 loss=total_loss / (i + 1),
#                 acc=total_correct / total_samples,
#                 lr=scheduler.get_last_lr()[0]
#             )
    
#     return total_loss / len(dataloader), total_correct / total_samples if total_samples > 0 else 0

# def validate(model, dataloader, criterion, device):
#     """
#     Validates the model (without Mixup).
#     """
#     model.eval()
#     total_loss = 0.0
#     all_labels = []
#     all_preds = []
    
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Validating"):
#             if batch is None: continue

#             video = batch['video'].to(device)
#             labels = batch['label'].to(device)
            
#             outputs = model(video)

#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
            
#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(predicted.cpu().numpy())

#     if not all_labels:
#         return 0.0, 0.0, "Validation set was empty or all samples failed to load."

#     val_loss = total_loss / len(dataloader)
#     val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
#     report = get_classification_report(all_labels, all_preds)
    
#     return val_loss, val_acc, report


# @hydra.main(version_base=None, config_path="../configs", config_name="vit")
# def main(config: DictConfig):
#     """
#     The main function to run the training pipeline.
#     """
#     logger.info("Starting training pipeline...")
#     logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

#     set_seed(config.seed)
#     device = torch.device(config.device if torch.cuda.is_available() else "cpu")
#     logger.info(f"Using device: {device}")

#     train_loader = create_dataloader(config, 'train')
#     val_loader = create_dataloader(config, 'val')

#     model = create_model(config).to(device)

#     optimizer = get_optimizer(model, config)
#     scheduler = get_scheduler(optimizer, config, len(train_loader))
    
#     if 'class_weights' in config and config.class_weights is not None:
#         class_weights = torch.tensor(config.class_weights, dtype=torch.float).to(device)
#         criterion = nn.CrossEntropyLoss(weight=class_weights)
#         logger.info(f"Applying class weights to loss function: {class_weights.tolist()}")
#     else:
#         criterion = nn.CrossEntropyLoss()
#         logger.info("Using unweighted CrossEntropyLoss.")

#     start_epoch, best_val_loss = load_checkpoint(model, optimizer, config, device)
    
#     epochs_no_improve = 0
#     patience = config.get('early_stopping_patience', 10)
#     logger.info(f"Using early stopping with patience of {patience} epochs.")

#     for epoch in range(start_epoch, config.epochs + 1):
#         logger.info(f"--- Epoch {epoch}/{config.epochs} ---")
        
#         train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
#         logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

#         val_loss, val_acc, report = validate(model, val_loader, criterion, device)
#         logger.info(f"Epoch {epoch} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
#         logger.info(f"Validation Classification Report:\n{report}")

#         is_best = val_loss < best_val_loss
#         if is_best:
#             best_val_loss = val_loss
#             epochs_no_improve = 0
#             save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, val_loss=val_loss, is_best=True, config=config)
#         else:
#             epochs_no_improve += 1
#             save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, val_loss=val_loss, is_best=False, config=config)

#         if epochs_no_improve >= patience:
#             logger.info(f"Validation loss has not improved for {patience} epochs. Stopping training early.")
#             break

#     logger.info("Training finished!")
#     logger.info(f"Best validation loss: {best_val_loss:.4f}")

# if __name__ == "__main__":
#     main()














# # import torch
# # import torch.nn as nn
# # from tqdm import tqdm
# # import hydra
# # from omegaconf import DictConfig, OmegaConf
# # import logging
# # import os
# # import numpy as np

# # from .dataloaders import create_dataloader
# # from .models import create_model
# # from .utils import set_seed, get_optimizer, get_scheduler, save_checkpoint, load_checkpoint, get_classification_report

# # # Setup logging
# # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# # logger = logging.getLogger(__name__)

# # def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, device):
# #     """
# #     Trains the model for one epoch.
# #     """
# #     model.train()
# #     total_loss = 0.0
# #     total_correct = 0
# #     total_samples = 0
    
# #     progress_bar = tqdm(dataloader, desc="Training")
# #     for i, batch in enumerate(progress_bar):
# #         if batch is None:
# #             logger.warning("Skipping a batch because all samples failed to load.")
# #             continue

# #         optimizer.zero_grad()
        
# #         video = batch['video'].to(device)
# #         labels = batch['label'].to(device)
        
# #         outputs = model(video)

# #         loss = criterion(outputs, labels)
# #         loss.backward()

# #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# #         optimizer.step()
# #         scheduler.step()

# #         total_loss += loss.item()
# #         _, predicted = torch.max(outputs.data, 1)
# #         total_correct += (predicted == labels).sum().item()
# #         total_samples += labels.size(0)

# #         if total_samples > 0:
# #             progress_bar.set_postfix(
# #                 loss=total_loss / (i + 1),
# #                 acc=total_correct / total_samples,
# #                 lr=scheduler.get_last_lr()[0]
# #             )
    
# #     return total_loss / len(dataloader), total_correct / total_samples if total_samples > 0 else 0

# # def validate(model, dataloader, criterion, device):
# #     """
# #     Validates the model and returns loss, accuracy, and a classification report.
# #     """
# #     model.eval()
# #     total_loss = 0.0
# #     all_labels = []
# #     all_preds = []
    
# #     with torch.no_grad():
# #         for batch in tqdm(dataloader, desc="Validating"):
# #             if batch is None: continue

# #             video = batch['video'].to(device)
# #             labels = batch['label'].to(device)
            
# #             outputs = model(video)

# #             loss = criterion(outputs, labels)
# #             total_loss += loss.item()
# #             _, predicted = torch.max(outputs.data, 1)
            
# #             all_labels.extend(labels.cpu().numpy())
# #             all_preds.extend(predicted.cpu().numpy())

# #     if not all_labels:
# #         return 0.0, 0.0, "Validation set was empty or all samples failed to load."

# #     val_loss = total_loss / len(dataloader)
# #     val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
# #     report = get_classification_report(all_labels, all_preds)
    
# #     return val_loss, val_acc, report


# # @hydra.main(version_base=None, config_path="../configs", config_name="vit")
# # def main(config: DictConfig):
# #     """
# #     The main function to run the training pipeline.
# #     """
# #     logger.info("Starting training pipeline...")
# #     logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")

# #     set_seed(config.seed)
# #     device = torch.device(config.device if torch.cuda.is_available() else "cpu")
# #     logger.info(f"Using device: {device}")

# #     train_loader = create_dataloader(config, 'train')
# #     val_loader = create_dataloader(config, 'val')

# #     model = create_model(config).to(device)

# #     optimizer = get_optimizer(model, config)
# #     scheduler = get_scheduler(optimizer, config, len(train_loader))
    
# #     if 'class_weights' in config and config.class_weights is not None:
# #         class_weights = torch.tensor(config.class_weights, dtype=torch.float).to(device)
# #         criterion = nn.CrossEntropyLoss(weight=class_weights)
# #         logger.info(f"Applying class weights to loss function: {class_weights.tolist()}")
# #     else:
# #         criterion = nn.CrossEntropyLoss()
# #         logger.info("Using unweighted CrossEntropyLoss.")

# #     start_epoch, best_val_loss = load_checkpoint(model, optimizer, config, device)
    
# #     epochs_no_improve = 0
# #     patience = config.get('early_stopping_patience', 10)
# #     logger.info(f"Using early stopping with patience of {patience} epochs.")

# #     for epoch in range(start_epoch, config.epochs + 1):
# #         logger.info(f"--- Epoch {epoch}/{config.epochs} ---")
        
# #         train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
# #         logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

# #         val_loss, val_acc, report = validate(model, val_loader, criterion, device)
# #         logger.info(f"Epoch {epoch} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
# #         logger.info(f"Validation Classification Report:\n{report}")

# #         is_best = val_loss < best_val_loss
# #         if is_best:
# #             best_val_loss = val_loss
# #             epochs_no_improve = 0
# #             # The 'is_best' argument is passed correctly here
# #             save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, val_loss=val_loss, is_best=True, config=config)
# #         else:
# #             epochs_no_improve += 1
# #             # --- BUG FIX: Corrected the function call to avoid the TypeError ---
# #             save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, val_loss=val_loss, is_best=False, config=config)

# #         if epochs_no_improve >= patience:
# #             logger.info(f"Validation loss has not improved for {patience} epochs. Stopping training early.")
# #             break

# #     logger.info("Training finished!")
# #     logger.info(f"Best validation loss: {best_val_loss:.4f}")

# # if __name__ == "__main__":
# #     main()