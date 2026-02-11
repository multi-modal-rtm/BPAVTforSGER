"""
Model definitions for the benchmark, modified to support attention extraction
and multi-task learning.

"""
import torch
import torch.nn as nn
import timm
from transformers import AutoModel, AutoConfig, TimesformerModel
import torch.nn.functional as F

class ViTAttentionHook:
    """
    A hook to capture the attention weights from the last block of a timm ViT.
    It attaches to the 'attn_drop' module and intercepts its *input*
    using a forward_pre_hook.
    """
    def __init__(self):
        self.attention_weights = None

    def __call__(self, module, input):

        try:
            if input and len(input) > 0:
                if isinstance(input[0], torch.Tensor):
                    self.attention_weights = input[0].detach() 
        except Exception as e:
            print(f"CRITICAL ERROR in hook __call__: {e}")

    def clear(self):
        self.attention_weights = None

class VideoModelWrapper(nn.Module):
    """
    A wrapper for various video models to standardize output and add classification heads.
    This version is set up for FULL FINE-TUNING.
    """
    def __init__(self, model_name, num_classes=4, num_frames=32, output_attentions=False):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.output_attentions = output_attentions
        self.attention_hook = None

        if "timesformer" in model_name:
            config = AutoConfig.from_pretrained(
                "facebook/timesformer-base-finetuned-k400", 
                output_attentions=output_attentions
            )
            self.backbone = TimesformerModel.from_pretrained(
                "facebook/timesformer-base-finetuned-k400",
                config=config
            )
            self.feature_dim = self.backbone.config.hidden_size
            
        elif "videomae" in model_name:
            config = AutoConfig.from_pretrained(
                "OpenGVLab/VideoMAEv2-Base", 
                output_attentions=output_attentions,
                trust_remote_code=True
            )
            self.backbone = AutoModel.from_pretrained(
                "OpenGVLab/VideoMAEv2-Base",
                config=config,
                trust_remote_code=True
            )
            self.feature_dim = 768 

            original_pos_embed = self.backbone.model.pos_embed 
            original_num_patches = original_pos_embed.shape[1] 
            original_num_frames = 8

            tubelet_size = 2 

            new_num_frames = self.num_frames // tubelet_size 
            
            num_patches_per_frame = (224 // 16) * (224 // 16) 
            new_num_patches = (new_num_frames * num_patches_per_frame) 
            
            if new_num_patches != original_num_patches:
                print(f"Interpolating VideoMAE pos_embed from {original_num_frames} frames to {new_num_frames} frames (Input: {self.num_frames}).")
                
                original_pos_embed_spatial = original_pos_embed.view(
                    1, original_num_frames, num_patches_per_frame, self.feature_dim
                )
                original_pos_embed_permuted = original_pos_embed_spatial.permute(0, 3, 1, 2)
                
                new_pos_embed = F.interpolate(
                    original_pos_embed_permuted,
                    size=(new_num_frames, num_patches_per_frame),
                    mode='bilinear',
                    align_corners=False,
                )
                
                new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
                new_pos_embed_flat = new_pos_embed.view(1, new_num_patches, self.feature_dim)
                self.backbone.model.pos_embed = nn.Parameter(new_pos_embed_flat)
                self.backbone.model.patch_embed.num_patches = new_num_patches
                self.backbone.model.patch_embed.num_frames = new_num_frames
            else:
                 print(f"VideoMAE pos_embed size already matches ({new_num_patches} patches). No interpolation needed.")
            dpr = 0.2
            
            for blk in self.backbone.model.blocks:
                if hasattr(blk, 'attn') and hasattr(blk.attn, 'attn_drop'):
                    blk.attn.attn_drop.p = 0.2
                if hasattr(blk, 'attn') and hasattr(blk.attn, 'proj_drop'):
                    blk.attn.proj_drop.p = 0.2 

                if hasattr(blk, 'drop_path'):
                    blk.drop_path.drop_prob = dpr
                    
            print(f"Applied aggressive regularization: Attn/Proj Dropout=0.1, DropPath Rate={dpr}")

            if self.output_attentions:
                print("Registering attention hook for videomae.")
                self.attention_hook = ViTAttentionHook()
                self.backbone.model.blocks[-1].attn.attn_drop.register_forward_pre_hook(self.attention_hook)

        elif "vit_base" in model_name or "swin_base" in model_name or "maxvit_base" in model_name or "mvitv2" in model_name:
            timm_model_name = {
                "vit_base": "vit_base_patch16_224.augreg_in21k_ft_in1k",
                "swin_base": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
                "maxvit_base": "maxvit_base_tf_224.in21k_ft_in1k",
                "mvitv2": 'mvitv2_base_cls.fb_inw21k'
            }[model_name]
            
            self.backbone = timm.create_model(
                timm_model_name,
                pretrained=True,
                num_classes=0
            )
            self.feature_dim = self.backbone.num_features
            
            if self.output_attentions:
                if "vit_base" in model_name:
                    print("Registering attention hook for vit_base.")
                    self.attention_hook = ViTAttentionHook()
                    self.backbone.blocks[-1].attn.attn_drop.register_forward_pre_hook(self.attention_hook)
                else:
                    print(f"Warning: Attention retrieval not implemented for {model_name}. Attentions will be None.")

        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
    def forward(self, x):
        b, t, c, h, w = x.shape
        x_reshaped = x.view(b * t, c, h, w)
        attentions = None
        
        if self.attention_hook:
            self.attention_hook.clear()

        if "timesformer" in self.model_name:
            outputs = self.backbone(x)
            features = outputs.last_hidden_state 
            if self.output_attentions:
                attentions = outputs.attentions
        
        elif "videomae" in self.model_name:
            x_permuted = x.permute(0, 2, 1, 3, 4)

            patch_features = self.backbone.model.patch_embed(x_permuted)

            pos_embed_on_device = self.backbone.model.pos_embed.to(patch_features.device)
            patch_features = patch_features + pos_embed_on_device

            for blk in self.backbone.model.blocks:
                patch_features = blk(patch_features)

            spatial_features = self.backbone.model.norm(patch_features)
            
            tubelet_size = 2 
            num_frames_in_model = t // tubelet_size
            num_patches_h = h // 16
            num_patches_w = w // 16
            
            features_spatial = spatial_features.view(
                b, 
                num_frames_in_model, 
                num_patches_h, 
                num_patches_w, 
                self.feature_dim
            )

            features_temporal = features_spatial.mean(dim=(2, 3))             
            features = torch.repeat_interleave(features_temporal, repeats=tubelet_size, dim=1).to(features_temporal.device)

            if self.output_attentions:
                attentions = self.attention_hook.attention_weights 

        elif "vit_base" in self.model_name or "swin_base" in self.model_name or "maxvit_base" in self.model_name or "mvitv2" in self.model_name:
            features_frames = self.backbone(x_reshaped) 
            features = features_frames.view(b, t, -1)
            
            if self.output_attentions:
                if "vit_base" in self.model_name:
                    attentions = self.attention_hook.attention_weights 
                else:
                    attentions = None 
        else:
            raise ValueError(f"Forward pass not implemented for {self.model_name}")

        logits = self.classifier(features)

        if self.output_attentions:
            return logits, attentions
        return logits

class AudioViTModel(nn.Module):
    def __init__(self, num_classes=4, num_frames=32):
        super().__init__()
        self.num_classes = num_classes
        pass
        
    def forward(self, x_video, x_audio):
        pass


def get_model(model_name, num_classes=4, num_frames=32, output_attentions=False):
    """
    Factory function to create the appropriate *single-task* model wrapper.
    """
    if model_name in ["vit_base", "swin_base", "maxvit_base", "timesformer", "videomae", "mvitv2"]:
        return VideoModelWrapper(model_name, num_classes, num_frames, output_attentions=output_attentions)
    elif model_name == "audiovit":
        return AudioViTModel(num_classes, num_frames)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


class MultiTaskModel(nn.Module):
    """
    A model that wraps a backbone and adds four separate classifier heads
    for simultaneous multi-task training.
    """
    def __init__(self, model_name, num_frames=32, output_attentions=False):
        super().__init__()
        self.model_name = model_name
        self.output_attentions = output_attentions

        self.base_model = get_model(
            model_name, 
            num_classes=4, 
            num_frames=num_frames,
            output_attentions=output_attentions
        )
        
        if hasattr(self.base_model, 'feature_dim'):
            feature_dim = self.base_model.feature_dim
        elif hasattr(self.base_model, 'classifier'):
             feature_dim = self.base_model.classifier.in_features
        else:
            raise ValueError("Could not determine feature dimension from base model")

        self.base_model.classifier = nn.Identity()

        self.dropout = nn.Dropout(p=0.5)
 
        self.classifier_boredom = nn.Linear(feature_dim, 4)
        self.classifier_engagement = nn.Linear(feature_dim, 4)
        self.classifier_confusion = nn.Linear(feature_dim, 4)
        self.classifier_frustration = nn.Linear(feature_dim, 4)

    def forward(self, x):
        
        attentions = None

        if self.output_attentions:
            base_output, attentions = self.base_model(x)
        else:
            base_output = self.base_model(x)

        features = base_output

        video_features = features.mean(dim=1) 
        
        video_features = self.dropout(video_features)

        logits_b = self.classifier_boredom(video_features)
        logits_e = self.classifier_engagement(video_features)
        logits_c = self.classifier_confusion(video_features)
        logits_f = self.classifier_frustration(video_features)
        
        logits_all = (logits_b, logits_e, logits_c, logits_f)
        
        if self.output_attentions:
            return logits_all, attentions
        
        return logits_all