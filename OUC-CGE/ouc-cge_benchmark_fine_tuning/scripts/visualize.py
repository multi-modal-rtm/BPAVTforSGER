import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
import sys
from tqdm import tqdm
import logging
from torchvision.transforms import v2 as T
import pandas as pd
import argparse  
import yaml 
from omegaconf import OmegaConf

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models import get_model
from src.visualization_hooks import ViTAttentionExtractor, VideoMAEAttentionExtractor 

logger = logging.getLogger(__name__)


def get_video_frames(video_path, config):
    """Loads and transforms frames from a single video path."""

    num_frames_to_sample = config.get('num_frames', 16) 
    
    abs_video_path = os.path.abspath(os.path.normpath(video_path))
    logger.info(f"Attempting to open video at absolute path: {abs_video_path}")
    
    if not os.path.exists(abs_video_path):
        logger.error(f"FATAL: File does not exist at path: {abs_video_path}")
        return None, None
    else:
        logger.info("File exists. Proceeding to open with OpenCV...")

    cap = cv2.VideoCapture(abs_video_path) 
    if not cap.isOpened():
        logger.error(f"Error: OpenCV cannot open video file {abs_video_path}")
        return None, None

    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if not frames:
        logger.warning(f"Warning: Could not read frames from {video_path}")
        return None, None

    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Resize(size=(224, 224), antialias=True),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    total_frames = len(frames)
    indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)
    sampled_frames = [frames[i] for i in indices]

    transformed_frames_list = [transform(frame) for frame in sampled_frames]
    
    transformed_frames_tensor = torch.stack(transformed_frames_list, dim=0) 

    video_tensor = transformed_frames_tensor.unsqueeze(0)

    original_frames = [cv2.resize(frames[i], (224, 224)) for i in indices]

    return video_tensor, original_frames

def create_attention_heatmap(attention_scores, frame_shape):
    """
    Converts raw attention scores into a visualization heatmap.
    """
    num_patches_side = int(attention_scores.shape[0]**0.5)
    if num_patches_side * num_patches_side != attention_scores.shape[0]:
        logger.warning(f"Attention scores ({attention_scores.shape[0]}) don't form a square grid. Visualization may be incorrect.")
        return np.zeros(frame_shape + (3,), dtype=np.uint8)
        
    attention_map = attention_scores.reshape(num_patches_side, num_patches_side).numpy(force=True)
    
    resized_map = cv2.resize(attention_map, frame_shape, interpolation=cv2.INTER_CUBIC)
    
    norm = Normalize(vmin=resized_map.min(), vmax=resized_map.max())
    cmap = plt.get_cmap('jet')
    colored_map = cmap(norm(resized_map))
    
    heatmap_bgr = cv2.cvtColor((colored_map * 255).astype(np.uint8), cv2.COLOR_RGBA2BGR)
    return heatmap_bgr

def overlay_heatmap(original_frame, heatmap, alpha=0.5):
    """Blends the heatmap with the original frame."""
    if len(original_frame.shape) == 3 and original_frame.shape[2] == 3:
        original_frame_bgr = cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR)
    else:
        original_frame_bgr = original_frame
        
    return cv2.addWeighted(heatmap, alpha, original_frame_bgr, 1 - alpha, 0)

def get_attention_extractor(model_name, model, device, config):
    """Factory function to select the correct attention extractor."""
    model_name_lower = model_name.lower()
    
    if model_name_lower == 'videomae':
        return VideoMAEAttentionExtractor(model, device, config)
    elif model_name_lower == 'vit':
        return ViTAttentionExtractor(model, device, config)

    else:
        logger.warning(f"No specific attention extractor found for model: {model_name}. Trying ViT extractor as a fallback.")
        try:
            return ViTAttentionExtractor(model, device, config)
        except Exception as e:
            raise ValueError(f"Could not initialize a fallback ViT extractor for {model_name}. Error: {e}")


def run_visualization(config):
    
    model_name = config.model_name
    logger.info(f"--- Starting Attention Visualization for {model_name} ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = os.path.join(config.results_dir, 'best_model.pth')
    
    logger.info(f"Loading best checkpoint from: {checkpoint_path}")
    
    model = get_model(config).to(device) 
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at {checkpoint_path}.")
        logger.error("Please ensure you have trained the model and the 'results_dir' in your config is correct.")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    logger.info("Model loaded successfully.")

    try:
        if not os.path.exists(config.root_dir):
            logger.error(f"Root directory not found: {config.root_dir}")
            return
            
        val_csv_path = os.path.join(config.root_dir, config.val_csv)
        if not os.path.exists(val_csv_path):
            logger.error(f"Validation CSV not found: {val_csv_path}")
            return

        logger.info(f"Loading validation CSV from: {val_csv_path}")

        if val_csv_path.endswith('.xlsx'):
            val_df = pd.read_excel(val_csv_path, header=None) 
            val_df.columns = ['video_path', 'label'] 
        else:
            val_df = pd.read_csv(val_csv_path, header=None, sep=' ')
            val_df.columns = ['video_path', 'label'] 
            
        video_col = 'video_path' 
        label_col = 'label'   
        
        if video_col not in val_df.columns or label_col not in val_df.columns:
            logger.error(f"Failed to parse CSV. Expected columns '{video_col}' and '{label_col}'.")
            return

        video_id_relative = val_df.iloc[19][video_col] # This is "videos/high/view1232.mp4"
        label_id = val_df.iloc[19][label_col]        # This is "2"
        
        if video_id_relative.startswith('videos/') or video_id_relative.startswith('videos\\'):
            video_id_relative = video_id_relative[7:] 

        video_path = os.path.join(config.root_dir, video_id_relative)
        video_path = os.path.normpath(video_path)
        
    except Exception as e:
        logger.error(f"Failed to load or parse validation file: {val_csv_path}")
        logger.error(f"Error: {e}")
        return
    
    logger.info(f"Loading sample video: {video_path} (Label: {label_id})")
    video_tensor, original_frames = get_video_frames(video_path, config)
    if video_tensor is None:
        return
        
    video_tensor = video_tensor.to(device)

    logger.info("Setting up attention extractor...")
    try:
        extractor = get_attention_extractor(model_name, model, device, config)
        extractor.register_hooks()
        
        logger.info("Performing forward pass to extract attention...")
        frame_attentions_tensor = extractor.get_attention(video_tensor)         
    except Exception as e:
        logger.error(f"Failed to extract attention: {e}")
        logger.exception("Full traceback:") 
        return
    finally:
        if 'extractor' in locals():
            extractor.remove_hooks() 
            
    logger.info("Attention extracted successfully.")

    output_filename = f'attention_visualization_{model_name}.mp4' 
    frame_h, frame_w = 224, 224
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_filename, fourcc, 5.0, (frame_w * 2, frame_h))
    logger.info(f"Creating output video: {output_filename}")
    
    num_frames = config.get('num_frames', 16)
    for i in range(num_frames): 
        original_frame_rgb = original_frames[i]
        
        attention_scores_for_frame = frame_attentions_tensor[i]
        
        heatmap_bgr = create_attention_heatmap(attention_scores_for_frame, (frame_w, frame_h))
        
        original_frame_bgr = cv2.cvtColor(original_frame_rgb, cv2.COLOR_RGB2BGR)
        
        overlaid_image = overlay_heatmap(original_frame_bgr, heatmap_bgr, alpha=0.6)

        cv2.putText(original_frame_bgr, f'Frame: {i+1} (Original)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(overlaid_image, f'Frame: {i+1} (Attention)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        comparison_image = np.hstack((original_frame_bgr, overlaid_image))
        
        out_video.write(comparison_image)
        
    out_video.release()
    logger.info(f"Successfully saved visualization to {output_filename}")
    logger.info(f"You can run this script for other models, e.g.: python scripts/visualize.py --config_name vit")

def main():
    parser = argparse.ArgumentParser(description="Run model attention visualization.")
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

        run_visualization(config)
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_file_path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.exception("Full traceback:")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
    main()