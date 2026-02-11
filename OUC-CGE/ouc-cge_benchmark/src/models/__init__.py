from .base_models import ViT, SwinT, SSAST, VideoMAE, MViT, MaxViT, TimeSformer
from .fusion_models import MBT

def get_model(config):
    """Selects and builds the model based on the configuration."""
    model_name = config['model_name'].lower()
    num_classes = 3 

    if model_name == 'vit':
        print("Building Vision Transformer (ViT) model.")
        return ViT(num_classes=num_classes)
    elif model_name == 'ssast':
        print("Building Spectrogram Transformer (SSAST) model.")
        return SSAST(num_classes=num_classes)
    elif model_name == 'swin':
        print("Building Swin Transformer model.")
        return SwinT(num_classes=num_classes)
    elif model_name == 'videomae':
        print("Building VideoMAE model.")
        return VideoMAE(num_classes=num_classes)
    elif model_name == 'mvit':
        print("Building Multiscale Vision Transformer (MViT) model.")
        return MViT(num_classes=num_classes)
    elif model_name == 'maxvit':
        print("Building MaxViT model.")
        return MaxViT(num_classes=num_classes)
    elif model_name == 'timesformer':
        print("Building TimeSformer model.")
        return TimeSformer(num_classes=num_classes)
    elif model_name == 'mbt':
        print("Building Multimodal Bottleneck Transformer (MBT) model.")
        return MBT(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
