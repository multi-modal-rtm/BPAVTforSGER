"""
Script to plot training and validation metrics from a training_log.csv file.
"""
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_logs(log_path):
    """
    Reads a training log CSV and generates plots for loss and F1 scores.
    """
    try:
        df = pd.read_csv(log_path)
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_path}")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if 'Epoch' not in df.columns:
        print("Error: Log file missing 'Epoch' column.")
        return

    # Create a directory for plots if it doesn't exist
    plot_dir = os.path.dirname(log_path)
    
    # --- 1. Plot Loss ---
    if 'Train_Loss' in df.columns and 'Val_Loss' in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df['Epoch'], df['Train_Loss'], label='Training Loss', marker='o', linestyle='--')
        plt.plot(df['Epoch'], df['Val_Loss'], label='Validation Loss', marker='x', linestyle='-')
        plt.title('Training and Validation Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Focal Loss)')
        plt.legend()
        plt.grid(True, linestyle=':')
        loss_plot_path = os.path.join(plot_dir, "plot_loss_dynamics.png")
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"Successfully saved loss plot to {loss_plot_path}")
    else:
        print("Skipping loss plot (missing required columns).")

    # --- 2. Plot F1 Scores ---
    acc_cols = ['Val_Acc_Avg', 'Val_Acc_Boredom','Val_Acc_Engagement','Val_Acc_Confusion','Val_Acc_Frustration']
    
    # Check if all required F1 columns exist
    if all(col in df.columns for col in acc_cols):
        plt.figure(figsize=(12, 7), dpi=400)
        
        # Plot the average F1 with a thick line
        plt.plot(df['Epoch'], df['Val_Acc_Avg'], label='Average Accuracy', marker='o', linewidth=3)
        
        # Plot individual task F1s with dashed lines
        plt.plot(df['Epoch'], df['Val_Acc_Boredom'], label='Val_Acc_Boredom', linestyle='--')
        plt.plot(df['Epoch'], df['Val_Acc_Engagement'], label='Val_Acc_Engagement', linestyle='--')
        plt.plot(df['Epoch'], df['Val_Acc_Confusion'], label='Val_Acc_Confusion', linestyle='--')
        plt.plot(df['Epoch'], df['Val_Acc_Frustration'], label='Val_Acc_Frustration', linestyle='--')

        plt.title('Accuracy Scores vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy Score')
        plt.legend(loc='best')
        plt.grid(True, linestyle=':')
        plt.ylim(0, 1.0) # F1 score is between 0 and 1
        
        f1_plot_path = os.path.join(plot_dir, "plot_acc_dynamics.png")
        plt.savefig(f1_plot_path)
        plt.close()
        print(f"Successfully saved accuracy plot to {f1_plot_path}")
    else:
        print("Skipping accuracy plot (missing required columns).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training dynamics from a log file.")
    parser.add_argument("--log_path", type=str, required=True,
                        help="Path to the training_log.csv file.")
    
    args = parser.parse_args()
    
    plot_logs(args.log_path)


# ```
# eof

# ### How to Use This:

# 1.  **Train your model** (e.g., VideoMAE) using the updated `train.py`.
#     ```bash
#     python src/train.py --config configs/videomae_config.yaml
#     ```
#     This will create a file at `results/videomae/training_log.csv`.

# 2.  **Run the new plotting script** on that log file.
#     ```bash
#     python src/plot_training_log.py --log_path "results/videomae/training_log.csv"