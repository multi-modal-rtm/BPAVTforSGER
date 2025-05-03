import torch
import torch.nn as nn

class CrossModalAdapterScratch(nn.Module):
    def __init__(self, video_dim=2048, audio_dim=768, adapter_dim=128, hidden_dim=512, num_classes=2, dropout=0.3):
        super(CrossModalAdapterScratch, self).__init__()

        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        # Cross-modal transformation
        self.cross_audio_to_video = nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, hidden_dim)
        )

        self.cross_video_to_audio = nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim),
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

    def forward(self, video_feat, audio_feat):
        '''
        video_feat: (B, T, 2048)
        audio_feat: (B, T, 768)
        '''
        v = self.video_proj(video_feat)
        a = self.audio_proj(audio_feat)

        v_to_a = self.cross_video_to_audio(v)
        a_to_v = self.cross_audio_to_video(a)

        x = torch.cat([v_to_a, a_to_v], dim=1)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        pooled = self.pool(x).squeeze(-1)
        return self.head(pooled)