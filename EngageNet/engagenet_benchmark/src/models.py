import torch
import torch.nn as nn
import timm
import logging
from transformers import AutoModel

logger = logging.getLogger(__name__)

class VideoModelWrapper(nn.Module):
    """
    A robust wrapper to adapt a timm image model for video classification.
    It freezes the backbone and trains only a new classification head.
    This version uses timm's helper functions to be compatible with more models.
    """
    def __init__(self, feature_extractor, num_features, num_classes):
        super().__init__()
        self.backbone = feature_extractor

        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)

        frame_features = self.backbone(x)

        video_features = frame_features.view(b, t, -1)

        clip_features = video_features.mean(dim=1)

        logits = self.head(clip_features)
        
        return logits

class TimeSformerWrapper(nn.Module):
    """
    A specific wrapper for the Hugging Face TimeSformer model that freezes the backbone.
    """
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        video_features = outputs.last_hidden_state[:, 0]
        return self.classifier(video_features)

class VideoMAEWrapper(nn.Module):
    """
    A specific wrapper for the Hugging Face VideoMAE model that freezes the backbone
    and handles the required tensor permutation.
    """
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 16, 224, 224)
            outputs = self.backbone(pixel_values=dummy_input)
            feature_dim = outputs.shape[-1]
        
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):

        pixel_values_permuted = x.permute(0, 2, 1, 3, 4)
        
        features = self.backbone(pixel_values=pixel_values_permuted)
        
        return self.classifier(features)


def create_model(config):
    """
    Factory function to create a video-only model based on the configuration.
    """
    model_name = config.model.name
    logger.info(f"Creating model: {model_name} with variant: {config.model.variant}")

    if model_name in ["ViT", "Swin", "MViT", "MaxViT"]:
        base_model = timm.create_model(
            config.model.variant,
            pretrained=True,
            num_classes=config.num_classes
        )

        classifier = base_model.get_classifier()
        num_features = classifier.in_features

        base_model.reset_classifier(0)

        for param in base_model.parameters():
            param.requires_grad = False
        
        model = VideoModelWrapper(
            feature_extractor=base_model,
            num_features=num_features,
            num_classes=config.num_classes
        )
        logger.info(f"Wrapped {model_name} and froze its backbone.")
    
    elif model_name == "TimeSformer":
        model = TimeSformerWrapper(
            model_name=config.model.variant,
            num_classes=config.num_classes
        )
        logger.info(f"Created Hugging Face TimeSformer model and froze its backbone.")

    elif model_name == "VideoMAE":
        model = VideoMAEWrapper(
            model_name=config.model.variant,
            num_classes=config.num_classes
        )
        logger.info(f"Created Hugging Face VideoMAE model and froze its backbone.")

    else:
        raise ValueError(f"Model '{model_name}' not recognized.")

    logger.info(f"Model {model_name} created successfully.")
    return model

