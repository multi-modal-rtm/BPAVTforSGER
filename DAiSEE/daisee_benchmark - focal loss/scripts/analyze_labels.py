import pandas as pd
from pathlib import Path

RAW_DATASET_ROOT = Path('D:\Abdulaziz\DAiSEE_Dataset_Raw\daisee_dataset\DAiSEE')
labels_file = RAW_DATASET_ROOT / 'Labels' / 'TrainLabels.csv'
df = pd.read_csv(labels_file)

label_columns = ['Boredom', 'Engagement', 'Confusion', 'Frustration ']

print("--- Label Distribution Analysis ---")

for col in label_columns:
    counts = df[col].value_counts().sort_index()
    print(f"\nDistribution for '{col.strip()}':")
    print(counts)

    weights = 1.0 / counts
    normalized_weights = weights / weights.sum()
    print(f"Calculated Weights for '{col.strip()}':")
    print(normalized_weights)