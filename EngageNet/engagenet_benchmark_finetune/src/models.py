import torch
import torch.nn as nn
import timm
from transformers import AutoModel, AutoConfig
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)

class VideoModelWrapper(nn.Module):
    """
    A robust wrapper for timm image models to handle video data.
    It separates the feature extractor from the classification head.
    """
    def __init__(self, feature_extractor, num_classes):
        super().__init__()
        self.backbone = feature_extractor

        num_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w) 

        features = self.backbone(x)

        video_features = features.view(b, t, -1).mean(dim=1)

        return self.head(video_features)


class TimeSformerWrapper(nn.Module):
    """Wrapper for the Hugging Face TimeSformer model."""
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)

        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        outputs = self.backbone(pixel_values=x)
        video_features = outputs.last_hidden_state[:, 0]
        return self.head(video_features)

class VideoMAEWrapper(nn.Module):
    """Wrapper for the Hugging Face VideoMAE model."""
    def __init__(self, model_name, num_classes):
        super().__init__()
        
        logger.info(f"Initializing VideoMAE with Standard Load (We will inject regularization manually)")

        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        self._apply_regularization(self.backbone, drop_path_rate=0.3, attn_drop_rate=0.3)

        with torch.no_grad():
            dummy_input = torch.randn(1, 16, 3, 224, 224) 
            dummy_input_permuted = dummy_input.permute(0, 2, 1, 3, 4)
            outputs = self.backbone(pixel_values=dummy_input_permuted)

            feature_dim = int(outputs.shape[-1])

        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(feature_dim, num_classes)
        )

    def _apply_regularization(self, model, drop_path_rate, attn_drop_rate):
        """
        Manually iterates through the model and sets drop rates on specific layers.
        This bypasses configuration issues with remote code.
        """
        logger.info(f"--- MANUALLY INJECTING REGULARIZATION ---")
        logger.info(f"Target Drop Path Rate: {drop_path_rate}")
        logger.info(f"Target Attention Drop Rate: {attn_drop_rate}")

        from timm.layers import DropPath
        drop_path_count = 0
        attn_drop_count = 0

        for name, module in model.named_modules():
            if isinstance(module, DropPath) or "DropPath" in module.__class__.__name__:
                module.drop_prob = drop_path_rate
                drop_path_count += 1

            if "Attention" in module.__class__.__name__:
                if hasattr(module, 'attn_drop') and isinstance(module.attn_drop, nn.Dropout):
                    module.attn_drop.p = attn_drop_rate
                    attn_drop_count += 1
                elif hasattr(module, 'attn_drop_rate'):
                    module.attn_drop_rate = attn_drop_rate
                    attn_drop_count += 1
                    
        logger.info(f"Updated {drop_path_count} DropPath layers to rate {drop_path_rate}")
        logger.info(f"Updated {attn_drop_count} Attention Dropout layers/params to rate {attn_drop_rate}")


    def forward(self, x):
       
        pixel_values_permuted = x.permute(0, 2, 1, 3, 4)

        video_features = self.backbone(pixel_values=pixel_values_permuted)
        
        return self.head(video_features)


def create_model(config: DictConfig) -> nn.Module:
    """
    Model factory to create the appropriate model based on the config.
    """
    model_name = config.model.name
    num_classes = config.num_classes
    
    # --- Check if we are in fine-tuning mode ---
    is_finetuning = config.model.get('finetune', False)
    
    variant = config.model.get('variant', None)
    if variant:
        logger.info(f"Creating model: {model_name} with variant: {variant}")
    else:
        logger.info(f"Creating model: {model_name}")

    if model_name == 'TimeSformer':
        model = TimeSformerWrapper(model_name=variant, num_classes=num_classes)
        
    elif model_name == 'VideoMAE':
        model = VideoMAEWrapper(model_name=variant, num_classes=num_classes)

    else: 
        base_model = timm.create_model(
            variant,
            pretrained=config.model.pretrained,
            num_classes=0
        )
        model = VideoModelWrapper(feature_extractor=base_model, num_classes=num_classes)
        
    if is_finetuning:
        logger.info(f"Fine-tuning mode enabled. Unfreezing backbone for {model_name}.")
        for param in model.backbone.parameters():
            param.requires_grad = True
    else:
        logger.info(f"Feature extraction mode. Freezing backbone for {model_name}.")
        for param in model.backbone.parameters():
            param.requires_grad = False
            
    logger.info(f"Model {model_name} created successfully.")
    return model