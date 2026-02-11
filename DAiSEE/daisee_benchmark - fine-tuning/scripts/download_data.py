import os
import kagglehub
from pathlib import Path


try:
    SCRIPT_DIR = Path(__file__).parent.resolve()
except NameError:
    SCRIPT_DIR = Path('.').resolve() / 'scripts'

PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'

def download_daisee_with_kagglehub(path_file_to_save):
    """
    Downloads and finds the path to the DAiSEE dataset using the kagglehub library.
    """
    print("--- Using kagglehub to download the DAiSEE dataset ---")
    print("Please ensure you are authenticated with Kaggle.")

    try:
        dataset_path = kagglehub.dataset_download("sannadbilal/daisee")
        print("\nDownload complete.")
        print(f"Dataset files are located at: {dataset_path}")

        with open(path_file_to_save, 'w') as f:
            f.write(str(dataset_path))
        print(f"SUCCESS: Saved dataset path to '{path_file_to_save}'")

    except Exception as e:
        print(f"\nAn error occurred during download.")
        print("Please ensure you have run 'pip install kagglehub' and are authenticated.")
        print(f"Error: {e}")
        exit()

if __name__ == "__main__":
    print(f"Project Root determined as: {PROJECT_ROOT}")
    print(f"Raw data directory set to: {RAW_DATA_DIR}")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    path_file = RAW_DATA_DIR / 'daisee_path.txt'

    download_daisee_with_kagglehub(path_file)