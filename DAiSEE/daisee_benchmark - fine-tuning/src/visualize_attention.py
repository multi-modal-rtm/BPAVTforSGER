"""
Script to visualize attention maps from a fine-tuned model.

(REVISED: Fixed VideoMAE key remapping and changed output to
 save a full attention video instead of a single frame.)
"""
import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm 

from models.models_attention import MultiTaskModel, get_model

NUM_FRAMES = 32
FRAME_SIZE = 224
LABEL_MAP = {
    "Boredom": 0,
    "Engagement": 1,
    "Confusion": 2,
    "Frustration": 3
}
CLASS_NAMES = {
    0: "Very Low",
    1: "Low",
    2: "High",
    3: "Very High"
}


def get_data_transforms():
    """Returns the same transforms as used in training."""
    return transforms.Compose([
        transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_video_frames(video_path, num_frames=NUM_FRAMES):
    """
    Loads, samples, and preprocesses a single video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = np.arange(num_frames) % total_frames

    frames = []
    original_frames = []
    transform = get_data_transforms()

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            if len(frames) > 0:
                frames.append(frames[-1])
                original_frames.append(original_frames[-1])
            else:
                continue 
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_frames.append(cv2.resize(frame_rgb, (FRAME_SIZE, FRAME_SIZE)))

        frame_pil = Image.fromarray(frame_rgb)
        processed_frame = transform(frame_pil)
        frames.append(processed_frame)

    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"Could not read any frames from {video_path}")

    video_tensor = torch.stack(frames)
    return video_tensor, original_frames


def process_attention(model_name, attentions, frame_index, num_heads=12):
    """
    Extracts and processes the attention map for a specific frame.
    """
    if attentions is None:
        raise ValueError(f"Error: Model {model_name} did not return attention maps.")

    if "timesformer" in model_name:
        attn_map = attentions[-1]

        attn_map = attn_map[0].mean(dim=0)

        raise NotImplementedError("Spatio-temporal attention processing for Timesformer is not yet implemented.")

    elif "videomae" in model_name:
        
        if attentions.dim() == 4 and attentions.shape[0] == 1:
            attn_map = attentions[0].mean(dim=0) 
        elif attentions.dim() == 3:
            attn_map = attentions.mean(dim=0) 
        else:
             raise ValueError(f"Unexpected VideoMAE attention shape: {attentions.shape}")
        
        if attn_map.shape[0] == 3136: 
            cls_attn = attn_map.mean(dim=0) 
        elif attn_map.shape[0] == 3137: 
            cls_attn = attn_map[0, 1:]
        else:
            raise ValueError(f"Unexpected attention map size: {attn_map.shape[0]}")
            
        num_frames_in_model = 16 
        num_patches_per_frame = 196 
        
        frame_attentions = cls_attn.view(num_frames_in_model, num_patches_per_frame) 
      
        frame_attentions_upsampled = torch.repeat_interleave(
            frame_attentions, 
            repeats=2, 
            dim=0
        )

        attn_map = frame_attentions_upsampled[frame_index] 

    elif "vit_base" in model_name:

        attn_map = attentions[frame_index]

        attn_map = attn_map.mean(dim=0)
        
    else:
        raise NotImplementedError(f"Attention processing not implemented for {model_name}")

    if "videomae" not in model_name:
        attn_map = attn_map[0, 1:] 

    num_patches = attn_map.shape[0]
    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"Cannot form square grid from {num_patches} patches.")
        
    attn_grid = attn_map.reshape(grid_size, grid_size).detach().cpu().numpy()

    heatmap = cv2.resize(attn_grid, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_CUBIC)

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    return heatmap

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {args.model_name}")
    model = MultiTaskModel(model_name=args.model_name, output_attentions=True)
    model.to(device)
    model.eval()

    print(f"Loading checkpoint from: {args.checkpoint_path}")

    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=args.weights_only)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            print("Checkpoint does not contain 'model_state_dict'. Trying to load root dict.")
            state_dict = checkpoint

        new_state_dict = {}
        key_mismatch_detected = False

        model_prefix = "base_model.backbone.model."
        
        for k, v in state_dict.items():
            new_key = k
            if k.startswith('module.'):
                new_key = k[len('module.'):]

            if (
                args.model_name == "videomae" 
                and new_key.startswith("base_model.backbone.") 
                and not new_key.startswith("base_model.backbone.model.")
            ):
                new_key = new_key.replace("base_model.backbone.", model_prefix, 1)
                key_mismatch_detected = True

            elif 'base_model.backbone.0.' in new_key:
                new_key = new_key.replace('base_model.backbone.0.', 'base_model.backbone.')
                key_mismatch_detected = True

            elif not new_key.startswith('base_model.') and not new_key.startswith('classifier_'):
                if new_key.startswith('backbone.'):
                     new_key = 'base_model.' + new_key
                else:
                     new_key = 'base_model.backbone.' + new_key
                key_mismatch_detected = True

            new_state_dict[new_key] = v

        if key_mismatch_detected:
            print("Warning: Checkpoint key mismatch detected.")
            print("Attempting to remap keys...")

        load_info = model.load_state_dict(new_state_dict, strict=False)
        
        missing_classifier_keys = [k for k in load_info.missing_keys if k.startswith('classifier_')]
        other_missing_keys = [k for k in load_info.missing_keys if not k.startswith('classifier_')]

        if missing_classifier_keys:
            print(f"Note: {len(missing_classifier_keys)} classifier keys were randomly initialized (this is expected).")
        if other_missing_keys:
            print("CRITICAL WARNING: The following non-classifier keys were MISSING from the checkpoint:")
            print(other_missing_keys)

        if load_info.unexpected_keys:
            print("Warning: Ignored unexpected keys in checkpoint (these are likely from an old checkpoint structure):")
            print(load_info.unexpected_keys[:10]) 
        
        if not other_missing_keys:
             print("Successfully loaded model weights.")
        else:
             print("Model loading may have FAILED. Check warnings.")
        
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    print(f"Loading video: {args.video_path}")
    try:
        video_tensor, original_frames = load_video_frames(args.video_path)
    except Exception as e:
        print(f"Error loading video: {e}")
        return
        
    video_tensor = video_tensor.unsqueeze(0)
    video_tensor = video_tensor.to(device)

    print("Running inference...")
    with torch.no_grad():
        (logits_b, logits_e, logits_c, logits_f), attentions = model(video_tensor)

    task_logits = {
        "Boredom": logits_b,
        "Engagement": logits_e,
        "Confusion": logits_c,
        "Frustration": logits_f
    }[args.task]

    video_logits = task_logits 
    prediction = torch.argmax(video_logits, dim=1).item()
    
    pred_class_name = CLASS_NAMES[prediction]
    print(f"Model prediction for {args.task}: {pred_class_name} (Class {prediction})")

    print(f"Processing attention and saving video for {args.model_name}...")

    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = f"{args.model_name}_{args.task}_pred_{prediction}.mp4"
    output_path = os.path.join(args.output_dir, output_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, 10.0, (FRAME_SIZE * 2, FRAME_SIZE))

    try:
        for i in tqdm(range(len(original_frames)), desc="Generating Video"):
            heatmap_norm = process_attention(
                args.model_name, 
                attentions, 
                frame_index=i
            )

            heatmap_bgr = (plt.cm.jet(heatmap_norm)[:, :, :3] * 255).astype(np.uint8)
            heatmap_bgr = cv2.cvtColor(heatmap_bgr, cv2.COLOR_RGB2BGR)

            original_frame_rgb = np.array(original_frames[i])
            original_frame_bgr = cv2.cvtColor(original_frame_rgb, cv2.COLOR_RGB2BGR)

            overlay = cv2.addWeighted(heatmap_bgr, 0.5, original_frame_bgr, 0.5, 0)

            comparison_image = np.hstack((original_frame_bgr, overlay))

            video_writer.write(comparison_image)

    except Exception as e:
        print(f"Error processing attention or writing video: {e}")
        video_writer.release() 
        return
    finally:
        video_writer.release()
    
    print(f"Successfully saved visualization to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model attention maps.")
    parser.add_argument("--model_name", type=str, required=True, 
                        choices=["vit_base", "swin_base", "maxvit_base", "timesformer", 
                                 "videomae", "mvitv2"],
                        help="Name of the model to visualize.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the trained .pth checkpoint file.")
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to the input video file.")
    parser.add_argument("--task", type=str, required=True, 
                        choices=["Boredom", "Engagement", "Confusion", "Frustration"],
                        help="The affective task to get a prediction for.")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                        help="Directory to save the output images.")

    parser.add_argument('--weights_only', action='store_true', default=False,
                        help='Use weights_only=True for torch.load() for security. Default: False.')

    args = parser.parse_args()
    
    if not args.weights_only:
        print("Warning: Loading checkpoint with weights_only=False. This is a security risk if the file is from an untrusted source.")
        print("Rerun with --weights_only=True to load safely.")
    
    main(args)