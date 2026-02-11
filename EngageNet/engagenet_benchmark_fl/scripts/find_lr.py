import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import logging
import os

from src.dataloaders import create_dataloader
from src.models import create_model
from src.utils import get_optimizer
from src.losses import get_criterion # Import our new unified loss function helper

from torch_lr_finder import LRFinder, TrainDataLoaderIter

logger = logging.getLogger(__name__)

class CustomDataLoaderIter(TrainDataLoaderIter):
    """
    A custom iterator that correctly unpacks data for video, audio,
    and audiovisual models.
    """
    def __init__(self, data_loader, modality='video'):
        super().__init__(data_loader)
        self.modality = modality

    def inputs_labels_from_batch(self, batch):
        if self.modality == 'audiovisual':
            return (batch["video"], batch["audio"]), batch["label"]
        elif self.modality == 'audio':
            return batch["audio"], batch["label"]
        else:
            return batch["video"], batch["label"]

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def find_learning_rate(config: DictConfig):
    """
    Runs the LR finder tool to help find an optimal learning rate.
    """
    model_name = config.model.name
    logger.info(f"--- Starting LR Finder for model: {model_name} ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_model(config).to(device)
    optimizer = get_optimizer(model, config)
    
    # Use our new helper function to create the correct loss function
    criterion = get_criterion(config, device)
    logger.info(f"Criterion ({config.loss_function}) created.")
    
    train_loader = create_dataloader(config, 'train')
    
    modality = config.get('modality', 'video')
    custom_train_iter = CustomDataLoaderIter(train_loader, modality=modality)

    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    logger.info("Running LR range test...")
    lr_finder.range_test(custom_train_iter, end_lr=1, num_iter=200, step_mode="exp")
    
    suggested_lr = lr_finder.suggestion()
    logger.info(f"LR Finder finished. Suggested LR: {suggested_lr:.2E}")
    
    output_dir = config.output_dir
    model_output_dir = os.path.join(output_dir, model_name)
    
    os.makedirs(model_output_dir, exist_ok=True)
    
    fig = lr_finder.plot()
    plot_path = os.path.join(model_output_dir, "lr_finder_plot.png")
    fig.savefig(plot_path)
    logger.info(f"Plot saved to {plot_path}")

    lr_finder.reset()

if __name__ == "__main__":
    find_learning_rate()

