import pandas as pd
import torch
from pathlib import Path

def calculate_class_weights(csv_path):
    """
    Calculates class weights based on inverse frequency for a given dataset.

    Args:
        csv_path (str): Path to the training CSV file (e.g., 'train_clean.csv').
                        The CSV should have two columns: path and label.
    """
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Calculating class weights from: {csv_path}")
     
    df = pd.read_csv(csv_path, header=None, sep=' ')
    labels = df.iloc[:, 1]
    
    class_counts = labels.value_counts().sort_index()
    
    if len(class_counts) == 0:
        print("No labels found in the CSV file.")
        return

    print("\nClass Distribution:")
    print(class_counts)

    num_classes = len(class_counts)
    total_samples = len(labels)
    
    weights = total_samples / (num_classes * class_counts)

    normalized_weights = (weights / weights.sum()) * num_classes
    
    print("\nCalculated Class Weights (for your YAML config):")

    print(f"class_weights: {normalized_weights.tolist()}")


if __name__ == '__main__':
    train_csv_file = 'D:/Abdulaziz/ouc-cge_benchmark/OUC-CGE/train.csv'
    calculate_class_weights(train_csv_file)
