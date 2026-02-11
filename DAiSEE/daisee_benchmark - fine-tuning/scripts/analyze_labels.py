import pandas as pd
from pathlib import Path

# Path to your raw dataset on your local machine
RAW_DATASET_ROOT = Path('D:/Abdulaziz/daisee_benchmark - fine-tuning/DAiSEE_Dataset_Raw/daisee_dataset/DAiSEE')
labels_file = RAW_DATASET_ROOT / 'Labels' / 'TrainLabels.csv'
df = pd.read_csv(labels_file)

# The label columns, including the trailing space for Frustration
label_columns = ['Boredom', 'Engagement', 'Confusion', 'Frustration ']

print("--- Label Distribution Analysis ---")

for col in label_columns:
    counts = df[col].value_counts().sort_index()
    print(f"\nDistribution for '{col.strip()}':")
    print(counts)
    
    # Calculate weights: weight = 1 / (number of samples in class)
    weights = 1.0 / counts
    # Normalize weights so they sum to 1 (optional but good practice)
    normalized_weights = weights / weights.sum()
    print(f"Calculated Weights for '{col.strip()}':")
    print(normalized_weights)