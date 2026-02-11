import hydra
from omegaconf import DictConfig
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging

logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="base_finetune", version_base=None)
def plot_curves(config: DictConfig):
    """
    Reads the training_log.csv file for a given model and
    generates a plot of its learning curves.
    """
    model_name = config.model.name
    model_output_dir = os.path.join(config.output_dir, model_name)
    log_file_path = os.path.join(model_output_dir, "training_log.csv")
    
    if not os.path.exists(log_file_path):
        logger.error(f"Log file not found at {log_file_path}")
        logger.error("Please run the training script first to generate the log file.")
        return

    logger.info(f"Loading log file from: {log_file_path}")
    try:
        df = pd.read_csv(log_file_path)
    except pd.errors.EmptyDataError:
        logger.error("Log file is empty. No data to plot.")
        return
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return

    if df.empty:
        logger.warning("Log file contains no data. Skipping plot generation.")
        return

    # Create a 2x1 plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- Plot 1: Loss ---
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue', linestyle='--')
    ax1.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='blue')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training and Validation Loss for {model_name}')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- Plot 2: Metrics (Accuracy and F1) ---
    ax2.plot(df['epoch'], df['train_acc'], label='Train Accuracy', color='green', linestyle='--')
    ax2.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', color='green')
    ax2.plot(df['epoch'], df['val_macro_f1'], label='Validation Macro F1-Score', color='red', linewidth=2)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Metric')
    ax2.set_title(f'Training and Validation Metrics for {model_name}')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Save the plot
    output_plot_path = os.path.join(model_output_dir, "learning_curve.png")
    plt.tight_layout()
    plt.savefig(output_plot_path)
    
    logger.info(f"Learning curve plot saved successfully to: {output_plot_path}")

if __name__ == "__main__":
    plot_curves()