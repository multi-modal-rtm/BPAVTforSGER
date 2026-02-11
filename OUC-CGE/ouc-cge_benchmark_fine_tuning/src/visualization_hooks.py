import torch
import torch.nn as nn
import numpy as np

class BaseAttentionExtractor:
    """A base class for attention extractors."""
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.captured_maps = []
        self.hook_handles = []

    def _get_attn_hook(self, module, input, output):
        """A simple hook to capture the first input of a module."""
        self.captured_maps.append(input[0].detach())

    def register_hooks(self):
        """Finds the target layer and registers the hook."""
        raise NotImplementedError("Each extractor must implement register_hooks.")

    def get_attention(self, video_tensor):
        """
        Runs the forward pass and processes the captured maps.
        Returns a tensor of shape (T, H_patch, W_patch)
        """
        raise NotImplementedError("Each extractor must implement get_attention.")

    def remove_hooks(self):
        """Removes all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.captured_maps = []

class ViTAttentionExtractor(BaseAttentionExtractor):
    """Attention extractor for a standard 2D ViT model."""
    
    def register_hooks(self):
        try:
            target_layer = self.model.backbone.blocks[-1].attn.attn_drop
            handle = target_layer.register_forward_hook(self._get_attn_hook)
            self.hook_handles.append(handle)
        except AttributeError:
            raise AttributeError("Failed to find 'backbone.blocks[-1].attn.attn_drop'. Is this a 'ViT' model?")
    
    def get_attention(self, video_tensor):
        b, t, c, h, w = video_tensor.shape
        video_tensor_reshaped = video_tensor.view(b * t, c, h, w)
        with torch.no_grad():
            _ = self.model.backbone.forward_features(video_tensor_reshaped)
        
        if not self.captured_maps:
            raise ValueError("Hook failed to capture any attention maps.")

        attentions = self.captured_maps[0] 

        att_map_per_frame = attentions.mean(dim=1)

        cls_att_per_frame = att_map_per_frame[:, 0, 1:] 

        num_patches_per_frame = cls_att_per_frame.shape[-1]
        cls_att_video = cls_att_per_frame.view(t, num_patches_per_frame)
        
        return cls_att_video.cpu() 

class VideoMAEAttentionExtractor(BaseAttentionExtractor):
    """Attention extractor for VideoMAE (3D patches, no CLS token)."""
    
    def register_hooks(self):
        try:
            self.target_layer = self.model.backbone.model.blocks[-1].attn.attn_drop
            handle = self.target_layer.register_forward_hook(self._get_attn_hook)
            self.hook_handles.append(handle)

            proj_kernel = self.model.backbone.model.patch_embed.proj.kernel_size
            self.tubelet_size = proj_kernel[0]
            self.patch_size_h = proj_kernel[1]
            self.patch_size_w = proj_kernel[2]

        except AttributeError:
            raise AttributeError("Failed to find layers for VideoMAE. Is this a 'VideoMAE' model?")

    def get_attention(self, video_tensor):

        pixel_values_permuted = video_tensor.permute(0, 2, 1, 3, 4)
        
        with torch.no_grad():
            _ = self.model.backbone(pixel_values=pixel_values_permuted)
            
        if not self.captured_maps:
            raise ValueError("Hook failed to capture any attention maps.")

        attentions = self.captured_maps[0] 

        att_map = attentions.squeeze(0).mean(dim=0) 

        avg_att_to_patches = att_map.mean(dim=0) 

        num_frames_in_model = self.config.num_frames // self.tubelet_size
        num_patches_per_frame = (224 // self.patch_size_h) * (224 // self.patch_size_w)
        
        if avg_att_to_patches.shape[0] != num_frames_in_model * num_patches_per_frame:
             raise ValueError(f"Attention dimension mismatch. Got {avg_att_to_patches.shape[0]}, expected {num_frames_in_model * num_patches_per_frame}.")
             
        frame_attentions = avg_att_to_patches.view(num_frames_in_model, num_patches_per_frame)

        frame_attentions_upsampled = torch.repeat_interleave(
            frame_attentions, 
            repeats=self.tubelet_size, 
            dim=0
        )
        
        return frame_attentions_upsampled.cpu() 