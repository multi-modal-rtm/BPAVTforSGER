import torch
import torch.nn as nn
import timm
from transformers import AutoModel, AutoConfig, AutoModelForVideoClassification
import torchvision.transforms.functional as F
from torchvision.transforms import v2 as T


def get_vit_model(num_outputs=16):
    """Loads a pre-trained ViT for fine-tuning."""
    print("Initializing ViT model (full fine-tuning)...")
    model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True, num_classes=0)
    
    return nn.Sequential(
        model,
        nn.Linear(model.num_features, num_outputs)
    )

class SwinForVideo(nn.Module):
    """A wrapper for the Swin Transformer for fine-tuning."""
    def __init__(self, num_classes=16):
        super().__init__()
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)

        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def get_swin_model(num_outputs=16):
    """Factory function that returns the SwinForVideo wrapper model."""
    print("Initializing Swin Transformer model (full fine-tuning)...")
    return SwinForVideo(num_outputs)

def get_mvit_model(num_outputs=16):
    """Loads a pre-trained MViT for fine-tuning."""
    print("Initializing MViTv2 model (full fine-tuning)...")
    model = timm.create_model('mvitv2_base_cls.fb_inw21k', pretrained=True)

    model.head.fc = nn.Linear(model.head.fc.in_features, num_outputs)
    return model

def get_maxvit_model(num_outputs=16):
    """Loads a pre-trained MaxViT for fine-tuning."""
    print("Initializing MaxViT model (full fine-tuning)...")
    model = timm.create_model('maxvit_base_tf_224.in1k', pretrained=True)

    model.head.fc = nn.Linear(model.head.fc.in_features, num_outputs)
    return model
    
class VideoMAE(nn.Module):
    """A robust wrapper for VideoMAEv2 for fine-tuning."""
    def __init__(self, num_labels=16):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "OpenGVLab/VideoMAEv2-Base", 
            trust_remote_code=True,
            use_safetensors=True
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 16, 224, 224)
            outputs = self.backbone(pixel_values=dummy_input)
            feature_dim = outputs.shape[-1]
        self.classifier = nn.Linear(feature_dim, num_labels)

    def forward(self, pixel_values):
        pixel_values_permuted = pixel_values.permute(0, 2, 1, 3, 4)
        features = self.backbone(pixel_values=pixel_values_permuted)
        return self.classifier(features)

def get_videomae_model(num_outputs=16):
    """Factory function that returns the CustomVideoMAE wrapper."""
    print("Initializing VideoMAE v2 model (full fine-tuning)...")
    return VideoMAE(num_labels=num_outputs)

class Timesformer(nn.Module):
    """A wrapper for Timesformer for fine-tuning."""
    def __init__(self, num_labels=16):
        super().__init__()
        self.model = AutoModelForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            use_safetensors=True
        )

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_labels)

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

def get_timesformer_model(num_outputs=16):
    """Factory function that returns the CustomTimesformer wrapper."""
    print("Initializing Timesformer model (full fine-tuning)...")
    return Timesformer(num_labels=num_outputs)