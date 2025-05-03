import torch
import torch.nn as nn
from transformers import AutoModel

class VATTPretrained(nn.Module):
    def __init__(self, video_model_name='MCG-NJU/vivit-video-classification', 
                 audio_model_name='facebook/wav2vec2-base', hidden_dim=512, 
                 num_classes=2, dropout=0.3):
        super(VATTPretrained, self).__init__()

        # Load pretrained video transformer (ViViT or similar)
        self.visual_encoder = AutoModel.from_pretrained(video_model_name)
        video_hidden = self.visual_encoder.config.hidden_size

        # Load pretrained audio transformer
        self.audio_encoder = AutoModel.from_pretrained(audio_model_name)
        audio_hidden = self.audio_encoder.config.hidden_size

        self.video_proj = nn.Linear(video_hidden, hidden_dim)
        self.audio_proj = nn.Linear(audio_hidden, hidden_dim)

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
        video_inputs: dict (tokenized video), e.g., {'pixel_values': tensor}
        audio_inputs: dict (tokenized audio), e.g., {'input_values': tensor}
        '''
        v_feat = self.visual_encoder(**video_inputs).last_hidden_state  # (B, T, Dv)
        a_feat = self.audio_encoder(**audio_inputs).last_hidden_state   # (B, T, Da)

        v = self.video_proj(v_feat)
        a = self.audio_proj(a_feat)

        x = torch.cat([v, a], dim=1)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        pooled = self.pool(x).squeeze(-1)
        return self.head(pooled)
