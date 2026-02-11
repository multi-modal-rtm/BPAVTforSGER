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
import pandas as pd # Added for robust CSV parsing

# Use explicit relative imports
from src.models import create_model
from src.dataloaders import EngageNetDataset

logger = logging.getLogger(__name__)

# --- Helper function to load and preprocess a single video ---

def get_video_frames(video_path, config):
    """Loads and transforms frames from a single video path."""
    
    abs_video_path = os.path.abspath(os.path.normpath(video_path))
    logger.info(f"Attempting to open video at absolute path: {abs_video_path}")
    
    if not os.path.exists(abs_video_path):
        logger.error(f"FATAL: File does not exist at path: {abs_video_path}")
        logger.error("Please manually check your file system to confirm this file exists.")
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

    # Use the same transforms as the validation set (minus augmentations)
    transform = T.Compose([
        T.ToImage(),  # Converts list of HWC numpy arrays to a CHW tensor
        T.ToDtype(torch.float32, scale=True),
        T.Resize(size=(224, 224), antialias=True),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Sample frames uniformly
    total_frames = len(frames)
    indices = np.linspace(0, total_frames - 1, config.num_frames, dtype=int)
    sampled_frames = [frames[i] for i in indices]
    
    transformed_frames_list = transform(sampled_frames) 
    
    # Stack the list of tensors into a single (T, C, H, W) tensor
    transformed_frames_tensor = torch.stack(transformed_frames_list, dim=0) 
    
    # Add batch dimension -> (B, T, C, H, W)
    video_tensor = transformed_frames_tensor.unsqueeze(0)
    
    # Also return the *original* untransformed frames for visualization
    original_frames = [cv2.resize(frames[i], (224, 224)) for i in indices]

    return video_tensor, original_frames

# --- Helper functions for visualization ---

def create_attention_heatmap(attention_scores, frame_shape):
    """
    Converts raw attention scores into a visualization heatmap.
    Args:
        attention_scores (torch.Tensor): 1D tensor of attention scores (e.g., shape [196]).
        frame_shape (tuple): (height, width) of the target frame.
    """
    # Reshape to 2D grid. The VideoMAE patch size is 16x16, so 224x224 -> 14x14 patches
    num_patches_side = int(attention_scores.shape[0]**0.5)
    if num_patches_side * num_patches_side != attention_scores.shape[0]:
        logger.warning("Attention scores don't form a square grid. Visualization may be incorrect.")
        return np.zeros(frame_shape + (3,), dtype=np.uint8)
        
    attention_map = attention_scores.reshape(num_patches_side, num_patches_side).numpy(force=True)
    
    # Resize the 14x14 map to the full frame size (224x224)
    resized_map = cv2.resize(attention_map, frame_shape, interpolation=cv2.INTER_CUBIC)
    
    # Normalize and colorize
    norm = Normalize(vmin=resized_map.min(), vmax=resized_map.max())
    cmap = plt.get_cmap('jet')
    colored_map = cmap(norm(resized_map))
    
    # Convert to 8-bit BGR (for OpenCV)
    heatmap_bgr = cv2.cvtColor((colored_map * 255).astype(np.uint8), cv2.COLOR_RGBA2BGR)
    return heatmap_bgr

def overlay_heatmap(original_frame, heatmap, alpha=0.5):
    """Blends the heatmap with the original frame."""
    # Ensure original_frame is BGR
    if len(original_frame.shape) == 3 and original_frame.shape[2] == 3:
         # Assuming input is RGB, convert to BGR for OpenCV
        original_frame_bgr = cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR)
    else:
        original_frame_bgr = original_frame
        
    return cv2.addWeighted(heatmap, alpha, original_frame_bgr, 1 - alpha, 0)

# --- Main visualization function ---

@hydra.main(config_path="../configs", config_name="videomae", version_base=None)
def visualize(config: DictConfig):
    
    logger.info("--- Starting Attention Visualization ---")
    
    # --- 1. Setup ---
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # This script is for VideoMAE only
    if config.model.name != 'VideoMAE':
        logger.error("This script is configured for VideoMAE. Please use config_name=videomae")
        return

    # --- 2. Load Model ---
    logger.info(f"Loading best checkpoint from: {config.output_dir}/{config.model.name}/best_checkpoint.pth")
    model = create_model(config).to(device)
    
    checkpoint_path = os.path.join(config.output_dir, config.model.name, 'best_checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")
        return
    
    # We must set weights_only=False because our checkpoint also contains
    # the optimizer state, epoch number, and F1-score, which are pickled.
    # We trust this file because we created it.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Model loaded successfully.")

    # --- 3. Get Sample Video ---
    # We'll just grab the first video from the validation set
    try:
        # Check the file extension to use the correct pandas function
        val_file_path = config.val_csv
        if val_file_path.endswith('.xlsx'):
            val_df = pd.read_excel(val_file_path, header=0)
        elif val_file_path.endswith('.csv'):
            val_df = pd.read_csv(val_file_path, header=0)
        else:
            raise ValueError(f"Unsupported validation file format: {val_file_path}")
            
        # Clean column names (e.g., remove leading spaces like in ' chunk')
        val_df.columns = [col.strip() for col in val_df.columns]
        
        # Find the video ID and label columns, handling 'chunk' or 'Video ID'
        video_col = 'Video ID' if 'Video ID' in val_df.columns else 'chunk'
        label_col = 'Label' if 'Label' in val_df.columns else 'label'

        video_id = val_df.iloc[0][video_col] # Get first video ID
        label = val_df.iloc[0][label_col]
        
        # The dataloader handles the label subdirectories.
        # For direct access, we just need the root Validation folder and the video_id.
        video_path = os.path.join(config.data_root, 'Validation', video_id)
        
        # Normalize the path to fix mixed forward/backward slashes (e.g., "D:/...\\...")
        video_path = os.path.normpath(video_path)
        
    except Exception as e:
        logger.error(f"Failed to load or parse validation file: {config.val_csv}")
        logger.error(f"Error: {e}")
        logger.error("Please ensure your validation file is a .csv or .xlsx and has columns named 'chunk' (or 'Video ID') and 'label' (or 'Label').")
        return
    
    logger.info(f"Loading sample video: {video_path} (Label: {label})")
    video_tensor, original_frames = get_video_frames(video_path, config)
    if video_tensor is None:
        return
        
    video_tensor = video_tensor.to(device)

    # --- 4. Extract Attention ---
    logger.info("Performing forward pass to extract attention...")
    
    attention_maps_capture = []
    def get_attn_probs_hook(module, input, output):
        # The input to the 'attn_drop' layer is the tensor of attention
        # probabilities right after the softmax.
        # 'input' is a tuple of args, so we take the first element.
        attention_maps_capture.append(input[0].detach())

    backbone = model.backbone
    
    try:
        # This path is based on the model structure you provided:
        # backbone.model.patch_embed.proj.kernel_size is (2, 16, 16)
        proj_kernel_size = backbone.model.patch_embed.proj.kernel_size
        tubelet_size = proj_kernel_size[0] # e.g., 2
        patch_size_h = proj_kernel_size[1] # e.g., 16
        patch_size_w = proj_kernel_size[2] # e.g., 16
        
        target_layer = backbone.model.blocks[-1].attn.attn_drop
        hook_handle = target_layer.register_forward_hook(get_attn_probs_hook)
    except AttributeError:
        logger.error("Failed to register hook or find patch_embed info. The model structure may have changed.")
        return

    # Permute from (B, T, C, H, W) to (B, C, T, H, W) for VideoMAE
    pixel_values_permuted = video_tensor.permute(0, 2, 1, 3, 4)
    
    with torch.no_grad():
        # Run the forward pass. The hook will fire automatically.
        outputs = backbone(
            pixel_values=pixel_values_permuted
        )

    # Remove the hook immediately after use
    hook_handle.remove()

    # Check if the hook successfully captured the attention
    if not attention_maps_capture:
        logger.error("Failed to capture attention maps. The hook was not successful.")
        return
        
    # `attentions` is the tensor captured by the hook
    # Shape: (B, NumHeads, SeqLen, SeqLen)
    attentions = attention_maps_capture[0]
    
    # Shape: (NumPatches, NumPatches). This model has NO [CLS] token.
    att_map = attentions.squeeze(0).mean(dim=0) 
    
    # Since there is no [CLS] token, we average the entire map (1568, 1568)
    # to get the attention *to* each patch *from* all other patches.
    avg_att_to_patches = att_map.mean(dim=0) # Shape: (1568,)
    
    
    # --- Use correct dimensions and upsample ---
    
    # Calculate the *model's* view of the patches
    num_frames_in_model = config.num_frames // tubelet_size # 16 // 2 = 8
    num_patches_h = 224 // patch_size_h # 224 // 16 = 14
    num_patches_w = 224 // patch_size_w # 224 // 16 = 14
    
    num_patches_per_frame = num_patches_h * num_patches_w # 14 * 14 = 196
    expected_total_patches = num_frames_in_model * num_patches_per_frame # 8 * 196 = 1568
    
    if avg_att_to_patches.shape[0] != expected_total_patches:
         logger.error(f"Attention dimension mismatch. Expected {expected_total_patches} patches, but got {avg_att_to_patches.shape[0]}.")
         return

    # Reshape to (T_model, NumPatchesPerFrame) -> (8, 196)
    frame_attentions = avg_att_to_patches.view(num_frames_in_model, num_patches_per_frame)
    
    # Upsample temporal dimension to match original video frames (8 -> 16)
    # We repeat each of the 8 attention maps `tubelet_size` (2) times.
    frame_attentions_upsampled = torch.repeat_interleave(
        frame_attentions, 
        repeats=tubelet_size, 
        dim=0
    )
    
    logger.info("Attention extracted and upsampled successfully.")

    # --- 5. Create Output Video ---
    output_filename = 'attention_video_output.mp4'
    frame_h, frame_w = 224, 224
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_filename, fourcc, 3.0, (frame_w * 2, frame_h))
    
    logger.info(f"Creating output video: {output_filename}")
    
    # Use config.num_frames (16) to loop over original frames
    for i in range(config.num_frames): 
        original_frame_rgb = original_frames[i]
        
        # --- THIS IS THE FIX ---
        # Use the upsampled attention tensor (which has 16 items)
        attention_scores_for_frame = frame_attentions_upsampled[i].cpu()
        
        heatmap_bgr = create_attention_heatmap(attention_scores_for_frame, (frame_w, frame_h))
        
        # Convert original frame to BGR for OpenCV functions
        original_frame_bgr = cv2.cvtColor(original_frame_rgb, cv2.COLOR_RGB2BGR)
        
        overlaid_image = overlay_heatmap(original_frame_bgr, heatmap_bgr, alpha=0.6)
        
        # Create a side-by-side comparison
        comparison_image = np.hstack((original_frame_bgr, overlaid_image))
        
        out_video.write(comparison_image)
        
    out_video.release()
    logger.info(f"Successfully saved visualization to {output_filename}")

if __name__ == "__main__":
    visualize()















# import hydra
# from omegaconf import DictConfig
# import torch
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
# import os
# from tqdm import tqdm
# import logging
# from torchvision.transforms import v2 as T
# import pandas as pd # Added for robust CSV parsing

# # Use explicit relative imports
# from src.models import create_model
# from src.dataloaders import EngageNetDataset

# logger = logging.getLogger(__name__)

# # --- Helper function to load and preprocess a single video ---

# def get_video_frames(video_path, config):
#     """Loads and transforms frames from a single video path."""
    
#     # --- NEW DEBUGGING BLOCK ---
#     # Get the absolute, normalized path to be 100% sure
#     abs_video_path = os.path.abspath(os.path.normpath(video_path))
#     logger.info(f"Attempting to open video at absolute path: {abs_video_path}")
    
#     # Explicitly check if the file exists
#     if not os.path.exists(abs_video_path):
#         logger.error(f"FATAL: File does not exist at path: {abs_video_path}")
#         logger.error("Please manually check your file system to confirm this file exists.")
#         return None, None
#     else:
#         logger.info("File exists. Proceeding to open with OpenCV...")
#     # --- END DEBUGGING BLOCK ---

#     cap = cv2.VideoCapture(abs_video_path) # Use the verified absolute path
#     if not cap.isOpened():
#         # This error will now mean OpenCV/FFmpeg has an issue, not that the file is missing
#         logger.error(f"Error: OpenCV cannot open video file {abs_video_path}")
#         return None, None

#     frames = []
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             # Convert BGR (OpenCV default) to RGB
#             frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     finally:
#         cap.release()

#     if not frames:
#         logger.warning(f"Warning: Could not read frames from {video_path}")
#         return None, None

#     # Use the same transforms as the validation set (minus augmentations)
#     transform = T.Compose([
#         T.ToImage(),  # Converts list of HWC numpy arrays to a CHW tensor
#         T.ToDtype(torch.float32, scale=True),
#         T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
#         T.CenterCrop(224),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

#     # Sample frames uniformly
#     total_frames = len(frames)
#     indices = np.linspace(0, total_frames - 1, config.num_frames, dtype=int)
#     sampled_frames = [frames[i] for i in indices]
    
#     # --- THIS IS THE FIX ---
#     # Apply transforms to each frame individually, returning a list of tensors
#     transformed_frames_list = transform(sampled_frames) # This is a list[Tensor(C,H,W)]
    
#     # Stack the list of tensors into a single (T, C, H, W) tensor
#     transformed_frames_tensor = torch.stack(transformed_frames_list, dim=0) 
    
#     # Add batch dimension -> (B, T, C, H, W)
#     video_tensor = transformed_frames_tensor.unsqueeze(0)
    
#     # Also return the *original* untransformed frames for visualization
#     original_frames = [cv2.resize(frames[i], (224, 224)) for i in indices]

#     return video_tensor, original_frames

# # --- Helper functions for visualization ---

# def create_attention_heatmap(attention_scores, frame_shape):
#     """
#     Converts raw attention scores into a visualization heatmap.
    
#     Args:
#         attention_scores (torch.Tensor): 1D tensor of attention scores (e.g., shape [196]).
#         frame_shape (tuple): (height, width) of the target frame.
#     """
#     # Reshape to 2D grid. The VideoMAE patch size is 16x16, so 224x224 -> 14x14 patches
#     num_patches_side = int(attention_scores.shape[0]**0.5)
#     if num_patches_side * num_patches_side != attention_scores.shape[0]:
#         logger.warning("Attention scores don't form a square grid. Visualization may be incorrect.")
#         return np.zeros(frame_shape + (3,), dtype=np.uint8)
        
#     attention_map = attention_scores.reshape(num_patches_side, num_patches_side).numpy(force=True)
    
#     # Resize the 14x14 map to the full frame size (224x224)
#     resized_map = cv2.resize(attention_map, frame_shape, interpolation=cv2.INTER_CUBIC)
    
#     # Normalize and colorize
#     norm = Normalize(vmin=resized_map.min(), vmax=resized_map.max())
#     cmap = plt.get_cmap('jet')
#     colored_map = cmap(norm(resized_map))
    
#     # Convert to 8-bit BGR (for OpenCV)
#     heatmap_bgr = cv2.cvtColor((colored_map * 255).astype(np.uint8), cv2.COLOR_RGBA2BGR)
#     return heatmap_bgr

# def overlay_heatmap(original_frame, heatmap, alpha=0.5):
#     """Blends the heatmap with the original frame."""
#     # Ensure original_frame is BGR
#     if len(original_frame.shape) == 3 and original_frame.shape[2] == 3:
#          # Assuming input is RGB, convert to BGR for OpenCV
#         original_frame_bgr = cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR)
#     else:
#         original_frame_bgr = original_frame
        
#     return cv2.addWeighted(heatmap, alpha, original_frame_bgr, 1 - alpha, 0)

# # --- Main visualization function ---

# @hydra.main(config_path="../configs", config_name="videomae", version_base=None)
# def visualize(config: DictConfig):
    
#     logger.info("--- Starting Attention Visualization ---")
    
#     # --- 1. Setup ---
#     device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
#     # This script is for VideoMAE only
#     if config.model.name != "VideoMAE":
#         logger.error("This script is configured for VideoMAE. Please use config_name=videomae")
#         return

#     # --- 2. Load Model ---
#     logger.info(f"Loading best checkpoint from: {config.output_dir}/{config.model.name}/best_checkpoint.pth")
#     model = create_model(config).to(device)
    
#     checkpoint_path = os.path.join(config.output_dir, config.model.name, 'best_checkpoint.pth')
#     if not os.path.exists(checkpoint_path):
#         logger.error(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")
#         return
    
#     # --- THIS IS THE FIX ---
#     # We must set weights_only=False because our checkpoint also contains
#     # the optimizer state, epoch number, and F1-score, which are pickled.
#     # We trust this file because we created it.
#     checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
#     logger.info("Model loaded successfully.")

#     # --- 3. Get Sample Video ---
#     # We'll just grab the first video from the validation set
#     try:
#         # --- THIS IS THE FIX ---
#         # Check the file extension to use the correct pandas function
#         val_file_path = config.val_csv
#         if val_file_path.endswith('.xlsx'):
#             val_df = pd.read_excel(val_file_path, header=0)
#         elif val_file_path.endswith('.csv'):
#             val_df = pd.read_csv(val_file_path, header=0)
#         else:
#             raise ValueError(f"Unsupported validation file format: {val_file_path}")
            
#         # Clean column names (e.g., remove leading spaces like in ' chunk')
#         val_df.columns = [col.strip() for col in val_df.columns]
        
#         # Find the video ID and label columns, handling 'chunk' or 'Video ID'
#         video_col = 'Video ID' if 'Video ID' in val_df.columns else 'chunk'
#         label_col = 'Label' if 'Label' in val_df.columns else 'label'

#         video_id = val_df.iloc[0][video_col] # Get first video ID
#         label = val_df.iloc[0][label_col]
        
#         # --- THIS IS THE FIX ---
#         # The dataloader handles the label subdirectories.
#         # For direct access, we just need the root Validation folder and the video_id.
#         video_path = os.path.join(config.data_root, 'Validation', video_id)
        
#         # Normalize the path to fix mixed forward/backward slashes (e.g., "D:/...\\...")
#         video_path = os.path.normpath(video_path)
        
#     except Exception as e:
#         logger.error(f"Failed to load or parse validation file: {config.val_csv}")
#         logger.error(f"Error: {e}")
#         logger.error("Please ensure your validation file is a .csv or .xlsx and has columns named 'chunk' (or 'Video ID') and 'label' (or 'Label').")
#         return
    
#     logger.info(f"Loading sample video: {video_path} (Label: {label})")
#     video_tensor, original_frames = get_video_frames(video_path, config)
#     if video_tensor is None:
#         return
        
#     video_tensor = video_tensor.to(device)

#     # --- 4. Extract Attention ---
#     logger.info("Performing forward pass to extract attention...")
    
#     # --- THIS IS THE FIX: We will use a forward hook on the correct layer ---
#     attention_maps_capture = []
#     def get_attn_probs_hook(module, input, output):
#         # The input to the 'attn_drop' layer is the tensor of attention
#         # probabilities right after the softmax.
#         # 'input' is a tuple of args, so we take the first element.
#         attention_maps_capture.append(input[0].detach())

#     backbone = model.backbone
    
#     # Register the hook on the last attention block
#     try:
#         # This path is based on the model structure you provided:
#         # backbone.model.patch_embed.proj.kernel_size is (2, 16, 16)
#         proj_kernel_size = backbone.model.patch_embed.proj.kernel_size
#         tubelet_size = proj_kernel_size[0] # e.g., 2
#         patch_size_h = proj_kernel_size[1] # e.g., 16
#         patch_size_w = proj_kernel_size[2] # e.g., 16
        
#         target_layer = backbone.model.blocks[-1].attn.attn_drop
#         hook_handle = target_layer.register_forward_hook(get_attn_probs_hook)
#     except AttributeError:
#         logger.error("Failed to register hook or find patch_embed info. The model structure may have changed.")
#         return

#     # Permute from (B, T, C, H, W) to (B, C, T, H, W) for VideoMAE
#     pixel_values_permuted = video_tensor.permute(0, 2, 1, 3, 4)
    
#     with torch.no_grad():
#         # Run the forward pass. The hook will fire automatically.
#         outputs = backbone(
#             pixel_values=pixel_values_permuted
#         )

#     # Remove the hook immediately after use
#     hook_handle.remove()

#     # Check if the hook successfully captured the attention
#     if not attention_maps_capture:
#         logger.error("Failed to capture attention maps. The hook was not successful.")
#         return
        
#     # `attentions` is the tensor captured by the hook
#     # Shape: (B, NumHeads, SeqLen, SeqLen)
#     attentions = attention_maps_capture[0]
    
#     # --- End of Fix ---

#     # The rest of the original logic for processing attention maps remains the same.
    
#     # We average the attention across all heads
#     mean_head_attention = torch.mean(attentions, dim=1).squeeze(0) # (SeqLen, SeqLen)
    
#     # The first token is the [CLS] token. We want the attention from this
#     # [CLS] token to all other *patch* tokens.
#     # SeqLen = 1 (CLS) + 16 (Temporal) * 14*14 (Spatial) = 3137
#     # This is complex. A simpler way is to look at the class attention.
#     # Let's use the average attention of all patch tokens to all other patch tokens.
    
#     # A common method is to average the attention maps from the last layer.
#     # We'll average over all heads.
#     # The shape is (B, Heads, NumPatches+1, NumPatches+1). +1 for [CLS] token.
#     # We will average the heads:
#     att_map = attentions.squeeze(0).mean(dim=0) # (NumPatches+1, NumPatches+1)
    
#     # We want the attention *to* the patch tokens, *from* the CLS token
#     # (or averaged from all tokens). Let's average "to" all patch tokens.
#     avg_att_to_patches = att_map.mean(dim=0) # (NumPatches)
    
#     # VideoMAE flattens time and space. NumPatches = T * H * W = 16 * 14 * 14 = 3136
#     # We need to reshape this back into frames.
#     num_frames_in_model = config.num_frames // tubelet_size # 16 // 2 = 8
#     num_patches_h = 224 // patch_size_h # 224 // 16 = 14
#     num_patches_w = 224 // patch_size_w # 224 // 16 = 14
    
#     num_patches_per_frame = num_patches_h * num_patches_w # 14 * 14 = 196
#     expected_total_patches = num_frames_in_model * num_patches_per_frame # 8 * 196 = 1568
    
#     if avg_att_to_patches.shape[0] != expected_total_patches:
#          logger.error(f"Attention dimension mismatch. Expected {expected_total_patches} patches, but got {avg_att_to_patches.shape[0]}.")
#          return

#     # Reshape to (T_model, NumPatchesPerFrame) -> (8, 196)
#     frame_attentions = avg_att_to_patches.view(num_frames_in_model, num_patches_per_frame)
    
#     # Upsample temporal dimension to match original video frames (8 -> 16)
#     # We repeat each of the 8 attention maps `tubelet_size` (2) times.
#     frame_attentions_upsampled = torch.repeat_interleave(
#         frame_attentions, 
#         repeats=tubelet_size, 
#         dim=0
#     )
#     logger.info("Attention extracted successfully.")

#     # --- 5. Create Output Video ---
#     output_filename = 'attention_video_output.mp4'
#     frame_h, frame_w = 224, 224
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out_video = cv2.VideoWriter(output_filename, fourcc, 3.0, (frame_w * 2, frame_h))
    
#     logger.info(f"Creating output video: {output_filename}")
    
#     for i in range(config.num_frames):
#         original_frame_rgb = original_frames[i]
#         attention_scores_for_frame = frame_attentions[i].cpu()
        
#         heatmap_bgr = create_attention_heatmap(attention_scores_for_frame, (frame_w, frame_h))
        
#         # Convert original frame to BGR for OpenCV functions
#         original_frame_bgr = cv2.cvtColor(original_frame_rgb, cv2.COLOR_RGB2BGR)
        
#         overlaid_image = overlay_heatmap(original_frame_bgr, heatmap_bgr, alpha=0.6)
        
#         # Create a side-by-side comparison
#         comparison_image = np.hstack((original_frame_bgr, overlaid_image))
        
#         out_video.write(comparison_image)
        
#     out_video.release()
#     logger.info(f"Successfully saved visualization to {output_filename}")

# if __name__ == "__main__":
#     visualize()

