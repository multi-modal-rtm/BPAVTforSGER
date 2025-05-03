import sys
import torch
import torch.nn as nn

# Add MBT repo to path (assume you cloned it in src/external/MBT)
sys.path.append("src/external/MBT")
from models.model import MBT  # official MBT class
from util.config import get_config

def load_mbt_with_custom_head(config_path, checkpoint_path, num_classes=2):
    config = get_config(config_path)
    model = MBT(config=config)

    # Load pretrained weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)  # classifier will mismatch

    # Replace final classifier
    input_dim = model.head.in_features
    model.head = nn.Linear(input_dim, num_classes)

    return model


# How to use
model = load_mbt_with_custom_head(
    config_path="src/external/MBT/configs/audioset.yaml",
    checkpoint_path="src/external/MBT/checkpoints/mbt_audioset.pth",
    num_classes=2
)

