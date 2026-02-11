import os
import pandas as pd
from decord import VideoReader, cpu
from tqdm import tqdm
import argparse

def clean_dataset_csv(root_dir, csv_filename, output_filename):
    """
    Reads a CSV file, checks each video for existence and readability,
    and writes the good entries to a new CSV file.
    """
    csv_path = os.path.join(root_dir, csv_filename)
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Loading CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path, header=None, sep=' ')
        df.columns = ['video_path', 'label']
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    good_entries = []
    print(f"Checking {len(df)} video files... This may take a while.")

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Cleaning {csv_filename}"):
        video_relative_path = row['video_path']
        label = row['label']

        if video_relative_path.startswith('videos/') or video_relative_path.startswith('videos\\'):
            video_relative_path = video_relative_path[7:]
            
        video_full_path = os.path.join(root_dir, video_relative_path)
        video_full_path = os.path.normpath(video_full_path)

        if not os.path.exists(video_full_path):
            continue

        try:
            vr = VideoReader(video_full_path, ctx=cpu(0))
            if len(vr) == 0:
                continue
        except Exception as e:
            continue

        good_entries.append({'video_path': row['video_path'], 'label': label})

    clean_df = pd.DataFrame(good_entries)
    output_path = os.path.join(root_dir, output_filename)

    clean_df.to_csv(output_path, sep=' ', header=False, index=False)
    
    print(f"\nCleaning complete for {csv_filename}.")
    print(f"Original entries: {len(df)}")
    print(f"Clean entries: {len(clean_df)}")
    print(f"Removed {len(df) - len(clean_df)} bad files.")
    print(f"Clean CSV saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Clean OUC-CGE dataset CSVs.")
    parser.add_argument(
        '--root_dir', 
        type=str, 
        required=True, 
        help="Path to the root of the OUC-CGE dataset (e.g., 'D:/.../OUC-CGE')"
    )
    args = parser.parse_args()

    clean_dataset_csv(args.root_dir, 'train.csv', 'train_clean.csv')

    clean_dataset_csv(args.root_dir, 'val.csv', 'val_clean.csv')

    clean_dataset_csv(args.root_dir, 'test.csv', 'test_clean.csv')
    
    print("\nAll dataset CSVs have been cleaned.")
    print("Please update your .yaml config files to use 'train_clean.csv', 'val_clean.csv', and 'test_clean.csv'")

if __name__ == "__main__":
    main()