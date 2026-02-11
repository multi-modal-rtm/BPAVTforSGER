import hydra
from omegaconf import DictConfig
import torch
import logging
import pandas as pd
from tqdm import tqdm
import numpy as np
import os

from .dataloaders import create_dataloader
from .models import create_model
from .utils import get_classification_report

# Set up logging
logger = logging.getLogger(__name__)

def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on a given dataset.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if batch is None:
                logger.warning("Skipping a batch because all samples failed to load.")
                continue

            video = batch['video']
            labels = batch['label']
            video, labels = video.to(device), labels.to(device)
            
            outputs = model(video)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    
    if not all_labels:
        return avg_loss, 0.0, "No samples were processed."

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    accuracy = (all_preds == all_labels).mean()
    report = get_classification_report(all_labels, all_preds)
    
    return avg_loss, accuracy, report


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(config: DictConfig) -> None:
    """
    Main function to run the evaluation on the test set.
    """
    output_dir = os.path.join(config.output_dir, config.model.name)
    logger.info(f"--- Starting Final Evaluation for {config.model.name} ---")
    logger.info(f"Output directory: {output_dir}")

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        test_loader = create_dataloader(config, 'test')
        logger.info(f"Test dataloader created with {len(test_loader.dataset)} samples.")
    except FileNotFoundError as e:
        logger.error(f"Could not create test dataloader: {e}")
        logger.error("Please ensure the 'test_csv' path in your config is correct.")
        return

    if not test_loader:
        logger.error("Test dataloader is empty. Exiting.")
        return

    model = create_model(config).to(device)

    checkpoint_path = os.path.join(output_dir, "best_checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        logger.error(f"Best checkpoint not found at {checkpoint_path}")
        logger.error("Please run the training script first to generate a checkpoint.")
        return

    logger.info(f"Loading best model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion = torch.nn.CrossEntropyLoss()

    test_loss, test_acc, report = evaluate(model, test_loader, criterion, device)

    logger.info("--- FINAL TEST SET RESULTS ---")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test Classification Report:\n{report}")


if __name__ == "__main__":
    main()