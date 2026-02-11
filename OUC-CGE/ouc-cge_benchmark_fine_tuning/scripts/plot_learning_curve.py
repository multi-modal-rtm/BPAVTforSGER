import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
import argparse 
import yaml 
import sys
from omegaconf import OmegaConf 

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

def plot_curves(config: OmegaConf):
    """
    Reads the training_log.csv file for a given model and
    generates a plot of its learning curves.
    """
    model_name = config.model_name

    log_file_path = os.path.join(config.results_dir, "training_log.csv")
    
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- Plot 1: Loss ---
    if 'train_loss' in df.columns and 'val_loss' in df.columns:
        ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue', linestyle='--')
        ax1.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='blue')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Training and Validation Loss for {model_name}')
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.6)
    else:
        logger.warning("Loss columns not found. Skipping loss plot.")

    # --- Plot 2: Metrics (Accuracy and F1) ---
    if 'val_acc' in df.columns and 'val_f1' in df.columns:
        if 'train_acc' in df.columns:
             ax2.plot(df['epoch'], df['train_acc'], label='Train Accuracy', color='green', linestyle='--')
             
        ax2.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', color='green')

        ax2.plot(df['epoch'], df['val_f1'], label='Validation Weighted F1-Score', color='red', linewidth=2)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Metric')
        ax2.set_title(f'Training and Validation Metrics for {model_name}')
        ax2.legend()
        ax2.grid(True, linestyle=':', alpha=0.6)
    else:
        logger.warning("Accuracy/F1 columns not found. Skipping metrics plot.")

    output_plot_path = os.path.join(config.results_dir, "learning_curve.png")
    plt.tight_layout()
    plt.savefig(output_plot_path)
    
    logger.info(f"Learning curve plot saved successfully to: {output_plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot learning curves from a training log.")
    parser.add_argument(
        '--config_name', 
        type=str, 
        required=True, 
        help="Name of the config file to use (e.g., 'videomae', 'vit')"
    )
    args = parser.parse_args()

    config_file_name = f"{args.config_name}.yaml"
    config_file_path = os.path.join(project_root, 'configs', config_file_name)

    if not os.path.exists(config_file_path):
        logger.error(f"Config file not found at: {config_file_path}")
        logger.error(f"Please make sure 'configs/{config_file_name}' exists.")
        return

    try:
        with open(config_file_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        config = OmegaConf.create(config_dict)

        plot_curves(config)
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_file_path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.exception("Full traceback:")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
    main()