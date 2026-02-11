import pandas as pd
import hydra
from omegaconf import DictConfig
import logging
import torch
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_and_process_labels(file_path):
    """Reads CSV or Excel, handles headers, filters irrelevant labels, and maps strings to integers."""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, header=0)
        else:
            raise ValueError("Unsupported label file format. Please use .csv or .xlsx")
    except FileNotFoundError:
        logger.error(f"Label file not found at: {file_path}")
        return None

    df.columns = [str(col).strip() for col in df.columns]
    
    # --- BUG FIX: Select the correct columns for video_id and label ---
    # Assuming the structure is [Index, VideoID, Label], we select the 2nd and 3rd columns.
    if df.shape[1] >= 3:
        logger.info(f"Detected {df.shape[1]} columns. Assuming Video ID is in column 2 and Label is in column 3.")
        df = df.iloc[:, [1, 2]]
    else:
        # Fallback for files that only have 2 columns
        df = df.iloc[:, [0, 1]]
        
    df.columns = ['video_id', 'engagement_level_str']

    # Filter out irrelevant classes
    irrelevant_labels = ['SNP(Subject Not Present)', 'label']
    initial_count = len(df)
    df = df[~df['engagement_level_str'].isin(irrelevant_labels)]
    filtered_count = len(df)
    if initial_count > filtered_count:
        logger.info(f"Filtered out {initial_count - filtered_count} irrelevant samples (SNP, header rows).")

    # Create a sorted mapping from unique string labels to integers
    unique_labels = sorted(df['engagement_level_str'].unique())
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    logger.info("--- Detected String Labels and Mapped to Integers ---")
    for label, index in label_map.items():
        logger.info(f"'{label}' -> {index}")

    df['engagement_level'] = df['engagement_level_str'].map(label_map)
    
    return df

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def analyze_weights(config: DictConfig):
    """
    Analyzes the class distribution in the training data and calculates
    inverse frequency weights for the loss function.
    """
    label_file = config.train_csv
    logger.info(f"Analyzing class weights for: {label_file}")

    df = read_and_process_labels(label_file)
    if df is None:
        return

    class_counts = df['engagement_level'].value_counts().sort_index()
    total_samples = len(df)
    
    logger.info("\n--- Class Distribution (after mapping and filtering) ---")
    for class_id, count in class_counts.items():
        percentage = (count / total_samples) * 100
        logger.info(f"Class {class_id}: {count} samples ({percentage:.2f}%)")
    logger.info(f"Total Samples: {total_samples}")

    class_weights = total_samples / (len(class_counts) * class_counts)
    weights_tensor = torch.tensor(class_weights.values, dtype=torch.float)
    
    logger.info("\n--- Calculated Class Weights for CrossEntropyLoss ---")
    logger.info(f"Weights Tensor: {weights_tensor.tolist()}")
    logger.info("To use these, add a 'class_weights' key to your config and pass them to the loss function:")
    logger.info("`criterion = nn.CrossEntropyLoss(weight=torch.tensor(config.class_weights).to(device))`")

if __name__ == "__main__":
    analyze_weights()
