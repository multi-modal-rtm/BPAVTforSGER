import os
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, WeightedRandomSampler

from .preprocess import extract_video, extract_audio, build_video_transform
from .configs import cfg

# Expected structure in configs/default.yaml:
# data:
#   roots:
#     daisee: /path/to/DAiSEE
#     ouc_cge: /path/to/OUC-CGE
#     ami: /path/to/AMI
#   manifests:
#     daisee: configs/daisee_manifest.csv
#     ouc_cge: configs/ouc_manifest.csv
#     ami: configs/ami_manifest.csv
#   batch_size: 8
#   num_workers: 4
# model:
#   num_classes: 4

class EngagementDataset(Dataset):
    """
    Wraps a single engagement dataset (DAiSEE, OUC-CGE, or AMI).
    Expects a CSV manifest with columns: video_path, audio_path, label, split.
    """
    def __init__(self, name: str, split: str, model_name: str):
        self.name = name
        manifest_path = cfg.data.manifests[name]
        df = pd.read_csv(manifest_path)
        self.rows = df[df['split'] == split].reset_index(drop=True)
        self.transform = build_video_transform(model_name)
        self.model_name = model_name
        self.root = cfg.data.roots[name]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        # Construct full paths
        video_path = os.path.join(self.root, row['video_path'])
        audio_path = os.path.join(self.root, row['audio_path'])
        # Extract features
        video = extract_video(video_path, self.model_name)
        audio = extract_audio(audio_path, self.model_name)
        # Spatial augmentations on video
        video = self.transform(video)
        # Label (assumed integer 0..num_classes-1)
        label = int(row['label'])
        return { 'video': video, 'audio': audio, 'label': label }


def make_balanced_weights(dataset: torch.utils.data.Dataset, n_classes: int) -> list:
    """
    Returns a list of per-sample weights so that each class is sampled equally.
    """
    counts = [0] * n_classes
    # First pass: count samples per class
    for sample in dataset:
        counts[sample['label']] += 1
    # Second pass: per-sample weight is inverse of class frequency
    weights = [1.0 / counts[sample['label']] for sample in dataset]
    return weights


def collate_fn(batch: list) -> dict:
    """
    Collate a list of samples into batch dict of stacked tensors.
    """
    videos = torch.stack([b['video'] for b in batch])          # [B, C, T, H, W]
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
    audios = None
    if batch[0]['audio'] is not None:
        audios = torch.stack([b['audio'] for b in batch])      # [B, 1 or feat_dim, ...]
    return { 'video': videos, 'audio': audios, 'label': labels }


def make_dataloader(split: str, model_name: str) -> DataLoader:
    """
    Build a concatenated DataLoader over DAiSEE, OUC-CGE, and AMI for a given split.
    Uses a weighted sampler for training to balance classes.
    """
    datasets = [ EngagementDataset(name, split, model_name)
                 for name in cfg.data.roots.keys() ]
    combined = ConcatDataset(datasets)

    sampler = None
    shuffle = False
    if split == 'train':
        weights = make_balanced_weights(combined, cfg.model.num_classes)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    else:
        shuffle = (split == 'val')

    loader = DataLoader(
        combined,
        batch_size=cfg.data.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return loader
