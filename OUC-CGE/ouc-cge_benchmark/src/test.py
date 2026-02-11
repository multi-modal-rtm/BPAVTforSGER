import os
import argparse
import torch
import numpy as np
import warnings

from utils import load_config, calculate_metrics
from dataloaders import create_dataloaders
from models import get_model 
from train import evaluate 

warnings.filterwarnings("ignore", category=UserWarning)

def test_model(experiment_dir):
    """
    Loads the best model from an experiment directory and evaluates it on the test set.
    """
    print(f"--- Starting Final Test Evaluation for Experiment: {experiment_dir} ---")

    config_path = os.path.join(experiment_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found in experiment directory: {config_path}")
    config = load_config(config_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    _, _, test_loader = create_dataloaders(config)
    if test_loader is None:
        print("No test data found. Exiting.")
        return

    model = get_model(config).to(device)
    model.config = config

    checkpoint_path = os.path.join(experiment_dir, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Best model checkpoint not found: {checkpoint_path}")
    
    print(f"Loading best model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    print("\nEvaluating on the test set...")
    labels, preds, probs = evaluate(model, test_loader, device)

    if len(labels) > 0:
        print("\n--- FINAL TEST SET RESULTS ---")
        _ = calculate_metrics(labels, preds, probs, class_names=['low', 'mid', 'high'])
    else:
        print("Evaluation failed. No samples were processed from the test set.")

    print("\n--- Test evaluation complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the best model from a training experiment on the test set.")
    parser.add_argument(
        '--experiment_dir', 
        type=str, 
        required=True, 
        help="Path to the experiment's results directory (e.g., 'results/vit_experiment')."
    )
    args = parser.parse_args()
    test_model(args.experiment_dir)