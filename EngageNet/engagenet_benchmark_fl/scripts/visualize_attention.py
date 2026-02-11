import hydra
from omegaconf import DictConfig
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
from tqdm import tqdm
import logging
from torchvision.transforms import v2 as T

from src.models import create_model
from src.dataloaders import EngageNetDataset

logger = logging.getLogger(__name__)

def get_video_frames(video_path, config):
    """Loads and transforms frames from a single video path."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Cannot open video file {video_path}")
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
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    total_frames = len(frames)
    indices = np.linspace(0, total_frames - 1, config.num_frames, dtype=int)
    sampled_frames = [frames[i] for i in indices]
    
    transformed_frames = transform(sampled_frames) 
    
    transformed_frames = transformed_frames.permute(1, 0, 2, 3) 

    video_tensor = transformed_frames.unsqueeze(0)
    
    original_frames = [cv2.resize(frames[i], (224, 224)) for i in indices]

    return video_tensor, original_frames

def create_attention_heatmap(attention_scores, frame_shape):
    """
    Converts raw attention scores into a visualization heatmap.
    
    Args:
        attention_scores (torch.Tensor): 1D tensor of attention scores (e.g., shape [196]).
        frame_shape (tuple): (height, width) of the target frame.
    """
    num_patches_side = int(attention_scores.shape[0]**0.5)
    if num_patches_side * num_patches_side != attention_scores.shape[0]:
        logger.warning("Attention scores don't form a square grid. Visualization may be incorrect.")
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

@hydra.main(config_path="../configs", config_name="videomae", version_base=None)
def visualize(config: DictConfig):
    
    logger.info("--- Starting Attention Visualization ---")

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    if config.model.name != "VideoMAE":
        logger.error("This script is configured for VideoMAE. Please use config_name=videomae")
        return

    logger.info(f"Loading best checkpoint from: {config.output_dir}/{config.model.name}/best_checkpoint.pth")
    model = create_model(config).to(device)
    
    checkpoint_path = os.path.join(config.output_dir, config.model.name, 'best_checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Model loaded successfully.")

    val_df = EngageNetDataset.load_label_file(config.val_csv)
    video_id = val_df.iloc[0]['Video ID'] 
    label = val_df.iloc[0]['Label']
    video_path = os.path.join(config.data_root, 'Validation', label, f"{video_id}.mp4")
    
    logger.info(f"Loading sample video: {video_path} (Label: {label})")
    video_tensor, original_frames = get_video_frames(video_path, config)
    if video_tensor is None:
        return
        
    video_tensor = video_tensor.to(device)

    logger.info("Performing forward pass to extract attention...")

    backbone = model.backbone

    pixel_values_permuted = video_tensor.permute(0, 2, 1, 3, 4)
    
    with torch.no_grad():
        outputs = backbone(
            pixel_values=pixel_values_permuted,
            output_attentions=True
        )
    attentions = outputs.attentions[-1] 

    att_map = attentions.squeeze(0).mean(dim=0) 

    avg_att_to_patches = att_map[1:, 1:].mean(dim=0)

    num_frames = config.num_frames
    num_patches_per_frame = (224 // 16) * (224 // 16) 
    
    if avg_att_to_patches.shape[0] != num_frames * num_patches_per_frame:
         logger.error("Attention dimension mismatch. Cannot reshape.")
         return

    frame_attentions = avg_att_to_patches.view(num_frames, num_patches_per_frame)
    logger.info("Attention extracted successfully.")

    output_filename = 'attention_video_output.mp4'
    frame_h, frame_w = 224, 224
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_filename, fourcc, 3.0, (frame_w * 2, frame_h))
    
    logger.info(f"Creating output video: {output_filename}")
    
    for i in range(num_frames):
        original_frame_rgb = original_frames[i]
        attention_scores_for_frame = frame_attentions[i].cpu()
        
        heatmap_bgr = create_attention_heatmap(attention_scores_for_frame, (frame_w, frame_h))

        original_frame_bgr = cv2.cvtColor(original_frame_rgb, cv2.COLOR_RGB2BGR)
        
        overlaid_image = overlay_heatmap(original_frame_bgr, heatmap_bgr, alpha=0.6)

        comparison_image = np.hstack((original_frame_bgr, overlaid_image))
        
        out_video.write(comparison_image)
        
    out_video.release()
    logger.info(f"Successfully saved visualization to {output_filename}")

if __name__ == "__main__":
    visualize()
