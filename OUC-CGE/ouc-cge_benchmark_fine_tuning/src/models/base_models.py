import torch.nn as nn
import timm
from transformers import AutoModel, AutoConfig

class ViT(nn.Module):
    """Standard Vision Transformer from timm, corrected for benchmarking."""
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True, num_classes=0)

        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x_reshaped = x.view(b * t, c, h, w)

        frame_features = self.backbone.forward_features(x_reshaped)[:, 0]

        video_features = frame_features.view(b, t, -1).mean(dim=1)
        
        return self.classifier(video_features)

class SwinT(nn.Module):
    """Swin Transformer from timm, corrected for benchmarking."""
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)

        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x_reshaped = x.view(b * t, c, h, w)

        frame_features = self.backbone(x_reshaped)

        video_features = frame_features.view(b, t, -1).mean(dim=1)
        
        return self.classifier(video_features)

class SSAST(nn.Module):
    """Audio Spectrogram Transformer from Hugging Face."""
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        output = self.backbone(input_values=x)
        features = output.last_hidden_state.mean(dim=1)
        return self.classifier(features)

class VideoMAE(nn.Module):
    """VideoMAE v2 from Hugging Face, corrected for benchmarking."""
    def __init__(self, num_classes=3):
        super().__init__()

        model_name = "OpenGVLab/VideoMAEv2-Base"

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        config.drop_rate = 0.3
        config.attn_drop_rate = 0.3

        self.backbone = AutoModel.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True) 

        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, x):
        pixel_values = x.permute(0, 2, 1, 3, 4)

        outputs = self.backbone(pixel_values=pixel_values)

        video_features = outputs
        
        return self.classifier(video_features)


class MViT(nn.Module):
    """Multiscale Vision Transformer (MViTv2) from timm."""
    def __init__(self, num_classes=3):
        super().__init__()

        self.backbone = timm.create_model('mvitv2_base.fb_in1k', pretrained=True, num_classes=0)
            
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x_reshaped = x.view(b * t, c, h, w)

        frame_features = self.backbone(x_reshaped)

        video_features = frame_features.view(b, t, -1).mean(dim=1)
        
        return self.classifier(video_features)


class MaxViT(nn.Module):
    """MaxViT from timm."""
    def __init__(self, num_classes=3):
        super().__init__()

        self.backbone = timm.create_model('maxvit_base_tf_224.in1k', pretrained=True, num_classes=0)
            
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x_reshaped = x.view(b * t, c, h, w)

        frame_features = self.backbone(x_reshaped)

        video_features = frame_features.view(b, t, -1).mean(dim=1)
        
        return self.classifier(video_features)
    

class TimeSformer(nn.Module):
    """TimeSformer from Hugging Face."""
    def __init__(self, num_classes=3):
        super().__init__()

        self.backbone = AutoModel.from_pretrained("facebook/timesformer-base-finetuned-k400")

        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        video_features = outputs.last_hidden_state[:, 0]       
        return self.classifier(video_features)
