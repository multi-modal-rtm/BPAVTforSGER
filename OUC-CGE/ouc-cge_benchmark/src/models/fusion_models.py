import torch
import torch.nn as nn
import timm
from transformers import AutoModel

class MBT(nn.Module):
    """Multimodal Bottleneck Transformer (MBT)."""
    def __init__(self, num_classes=3, vision_dim=768, audio_dim=768, bottleneck_dim=256):
        super().__init__()
        self.vision_backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        for p in self.vision_backbone.parameters(): p.requires_grad = False
        
        self.audio_backbone = AutoModel.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')
        for p in self.audio_backbone.parameters(): p.requires_grad = False

        self.bottleneck_tokens = nn.Parameter(torch.randn(1, 1, bottleneck_dim))
        self.vision_to_bottleneck = nn.Linear(vision_dim, bottleneck_dim)
        self.audio_to_bottleneck = nn.Linear(audio_dim, bottleneck_dim)
        
        self.fusion_attention = nn.MultiheadAttention(embed_dim=bottleneck_dim, num_heads=8, batch_first=True)
        self.classifier = nn.Linear(bottleneck_dim, num_classes)

    def forward(self, video, audio):
        b, t, c, h, w = video.shape
        video_reshaped = video.view(b * t, c, h, w)
        vision_features = self.vision_backbone.forward_features(video_reshaped)[:, 0]
        vision_features = vision_features.view(b, t, -1).mean(dim=1)

        audio_output = self.audio_backbone(input_values=audio)
        audio_features = audio_output.last_hidden_state.mean(dim=1)
        
        vision_proj = self.vision_to_bottleneck(vision_features).unsqueeze(1)
        audio_proj = self.audio_to_bottleneck(audio_features).unsqueeze(1)
        
        modalities = torch.cat([vision_proj, audio_proj], dim=1)
        bottleneck = self.bottleneck_tokens.expand(b, -1, -1)
        
        fused_features, _ = self.fusion_attention(query=bottleneck, key=modalities, value=modalities)
        
        return self.classifier(fused_features.squeeze(1))