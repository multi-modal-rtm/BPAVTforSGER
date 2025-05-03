import torch
import torch.nn as nn
from transformers import AutoModel

class CrossModalAdapter(nn.Module):
    def __init__(self, video_model_name='google/vit-base-patch16-224-in21k',
                 audio_model_name='facebook/wav2vec2-base',
                 adapter_dim=128, hidden_dim=512, num_classes=2, dropout=0.3):
        super(CrossModalAdapter, self).__init__()

        # Frozen backbones
        self.visual_encoder = AutoModel.from_pretrained(video_model_name)
        self.audio_encoder = AutoModel.from_pretrained(audio_model_name)

        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        video_hidden = self.visual_encoder.config.hidden_size
        audio_hidden = self.audio_encoder.config.hidden_size

        # Cross-modal adapter: audio attends to video + vice versa
        self.cross_video_to_audio = nn.Sequential(
            nn.Linear(video_hidden, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, hidden_dim)
        )

        self.cross_audio_to_video = nn.Sequential(
            nn.Linear(audio_hidden, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, hidden_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=1024, 
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, video_inputs, audio_inputs):
        '''
        video_inputs: dict for ViT (e.g., {'pixel_values': ...})
        audio_inputs: dict for Wav2Vec2 (e.g., {'input_values': ...})
        '''
        with torch.no_grad():
            v_feat = self.visual_encoder(**video_inputs).last_hidden_state  # (B, T, Dv)
            a_feat = self.audio_encoder(**audio_inputs).last_hidden_state   # (B, T, Da)

        # Cross-modal transformation
        a_to_v = self.cross_audio_to_video(a_feat)  # audio informs video
        v_to_a = self.cross_video_to_audio(v_feat)  # video informs audio

        x = torch.cat([a_to_v, v_to_a], dim=1)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        pooled = self.pool(x).squeeze(-1)
        return self.head(pooled)