import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from .configs import cfg

# ─── Backbone loader functions ──────────────────────────────────────────────

def load_mbt():
    """
    Multimodal Bottleneck Transformer (MBT).
    Requires Scenic/JAX implementation; adjust import path as needed.
    """
    from scenic.projects.vatt.model import MBTModel
    return MBTModel()

def load_vatt():
    """
    Video-Audio-Text Transformer (VATT).
    Placeholder: replace with your VATT implementation import.
    """
    from .vatt import VATTModel
    return VATTModel()

def load_avt():
    """
    Audio-Video Transformer (AVT).
    Placeholder: replace with your AVT implementation import.
    """
    from .avt import AVTModel
    return AVTModel()

def load_swin():
    """
    Video Swin Transformer.
    Example import; adjust to your Swin implementation.
    """
    from swin_transformer import SwinTransformer
    return SwinTransformer()

# Map model names to loader functions
MODEL_BACKBONES = {
    "mbt":  load_mbt,
    "vatt": load_vatt,
    "avt":  load_avt,
    "swin": load_swin,
}

# ─── Pretrained checkpoint loader ────────────────────────────────────────────

def load_pretrained(model_name: str):
    """
    Instantiate backbone and optionally load Kinetics-400 pretrained weights.
    """
    # 1. Instantiate
    model = MODEL_BACKBONES[model_name]()

    # 2. Load pretrained if requested
    if cfg.model.pretrained:
        # Assumes a repo named '<model_name>-k400' hosting a 'pytorch_model.bin'
        repo_id = f"{model_name}-k400"
        filename = "pytorch_model.bin"
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=False)

    return model

# ─── Engagement recognition network ────────────────────────────────────────

class EngagementNet(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        # Load backbone (with pretrained weights if available)
        self.backbone = load_pretrained(model_name)

        # Infer feature dimension
        if hasattr(self.backbone, "embed_dim"):
            feat_dim = self.backbone.embed_dim
        elif hasattr(self.backbone, "out_features"):
            feat_dim = self.backbone.out_features
        else:
            # Fallback: specify in your config
            feat_dim = cfg.model.backbone_feat_dim

        # Classification head
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, video: torch.Tensor, audio: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
          video: Tensor of shape [B, C, T, H, W]
          audio: Tensor [B, F, T_audio] or None

        Returns:
          logits: Tensor [B, num_classes]
        """
        # Some backbones accept (video, audio), others only video
        try:
            feats = self.backbone(video, audio)
        except TypeError:
            feats = self.backbone(video)

        logits = self.classifier(feats)
        return logits
