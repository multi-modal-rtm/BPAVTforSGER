import torch
import torch.nn as nn
import timm
from transformers import AutoModel, AutoConfig, AutoModelForVideoClassification
import torchvision.transforms.functional as F
from torchvision.transforms import v2 as T

def get_vit_model(num_outputs=16):
    """Loads a pre-trained ViT with a frozen backbone."""
    model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.head = nn.Linear(model.head.in_features, num_outputs)
    return model

class SwinTransformer(nn.Module):
    """A wrapper for the Swin Transformer with a frozen backbone."""
    def __init__(self, num_classes=16):
        super().__init__()
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def get_swin_model(num_outputs=16):
    """Factory function that returns the SwinForVideo wrapper model."""
    print("Initializing Swin Transformer model (frozen backbone)...")
    return SwinTransformer(num_outputs)

def get_mvit_model(num_outputs=16):
    """Loads a pre-trained MViT with a frozen backbone."""
    model = timm.create_model('mvitv2_base.fb_in1k', pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.head.fc = nn.Linear(model.head.fc.in_features, num_outputs)
    return model

def get_maxvit_model(num_outputs=16):
    """Loads a pre-trained MaxViT with a frozen backbone."""
    model = timm.create_model('maxvit_base_tf_224.in1k', pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.head.fc = nn.Linear(model.head.fc.in_features, num_outputs)
    return model

class VideoMAE(nn.Module):
    """
    A robust wrapper for VideoMAEv2 that handles tensor permutation
    and separates the backbone from the classifier for fine-tuning.
    """
    def __init__(self, num_labels=16):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "OpenGVLab/VideoMAEv2-Base",
            trust_remote_code=True
        )
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 16, 224, 224)
            outputs = self.backbone(pixel_values=dummy_input)
            feature_dim = outputs.shape[-1]
        
        print(f"Detected VideoMAE feature dimension: {feature_dim}")

        self.classifier = nn.Linear(feature_dim, num_labels)

    def forward(self, pixel_values):
        pixel_values_permuted = pixel_values.permute(0, 2, 1, 3, 4)
        
        features = self.backbone(pixel_values=pixel_values_permuted)
        
        return self.classifier(features)

def get_videomae_model(num_outputs=16):
    """Loads a pre-trained VideoMAEv2 for fine-tuning using a robust wrapper."""
    print("Initializing VideoMAE v2 model (fine-tuning enabled)...")
    return VideoMAE(num_labels=num_outputs)


class Timesformer(nn.Module):
    """A wrapper for Timesformer that handles tensor permutation and frozen backbone."""
    def __init__(self, num_labels=16):
        super().__init__()
        self.model = AutoModelForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            use_safetensors=True
        )
        for param in self.model.timesformer.parameters():
            param.requires_grad = False
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_labels)

    def forward(self, pixel_values):
        pixel_values_permuted = pixel_values
        outputs = self.model(pixel_values=pixel_values_permuted)
        return outputs.logits

def get_timesformer_model(num_outputs=16):
    """Factory function that returns the CustomTimesformer wrapper."""
    print("Initializing Timesformer model (frozen backbone)...")
    return Timesformer(num_labels=num_outputs)