import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchvision.transforms import v2 as T
from decord import VideoReader, cpu
import numpy as np
from pathlib import Path


def collate_fn_raw_video(batch):
    """Custom collate function for the raw video dataset."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        # Return empty tensors if the whole batch is invalid
        return torch.tensor([]), torch.tensor([])
    
    videos = torch.stack([item['video'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    return videos, labels

class DAiSEERawVideoDataset(Dataset):
    """Loads DAiSEE video frames directly from video files on-the-fly."""
    def __init__(self, split, config):
        self.config = config
        self.raw_dataset_root = Path(config['data']['raw_data_root'])
        labels_file = self.raw_dataset_root / 'Labels' / f'{split}Labels.csv'
        self.annotations = pd.read_csv(labels_file)

        self.label_columns = ['Boredom', 'Engagement', 'Confusion', 'Frustration ']
        self.num_frames = config['training'].get('num_frames', 16)

        self.transform = T.Compose([
            T.ToDtype(torch.float32, scale=True),
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        try:
            video_info = self.annotations.iloc[index]
            clip_id = Path(video_info['ClipID']).stem
            
            video_path_search = list((self.raw_dataset_root / 'DataSet').rglob(f"{clip_id}.*"))
            if not video_path_search:
                 return None
            video_path = video_path_search[0]

            vr = VideoReader(str(video_path), ctx=cpu(0))
            if len(vr) < self.num_frames: return None
            
            total_frames = len(vr)
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = vr.get_batch(frame_indices).asnumpy()
            
            video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)
            video_tensor = self.transform(video_tensor)

            labels = video_info[self.label_columns].values.astype(np.int64)
            labels_tensor = torch.from_numpy(labels)

            return {"video": video_tensor, "label": labels_tensor}
        except Exception as e:
            return None


class DAiSEEProcessedDataset(Dataset):
    """Loads pre-processed audio (.wav) and video frames (.png sequences)."""
    def __init__(self, split, config):
        self.split = split
        self.config = config
        self.processed_data_root = Path(config['data']['processed_data_root'])

        self.raw_dataset_root = Path('D:/Abdulaziz/daisee_benchmark/DAiSEE_Dataset_Raw/daisee_dataset/DAiSEE')
        
        self.data_mode = config['data'].get('data_mode', 'audiovisual')
        self.label_columns = ['Boredom', 'Engagement', 'Confusion', 'Frustration ']
        
        if self.data_mode != 'audio_only':
            self.video_transforms = self._get_video_transforms()
        if self.data_mode != 'video_only':
            self.audio_transforms = self._get_audio_transforms()
            
        self.samples = self._create_sample_list()

    def _create_sample_list(self):
        print(f"[{self.split}] Creating sample list for processed data...")
        labels_file = self.raw_dataset_root / 'Labels' / f'{self.split}Labels.csv'
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found at {labels_file}")
        labels_df = pd.read_csv(labels_file)
        
        samples = []
        video_sequence_root = self.processed_data_root / self.split / 'video'
        audio_root = self.processed_data_root / self.split / 'audio'

        for _, row in labels_df.iterrows():
            clip_id = Path(row['ClipID']).stem

            audio_path = audio_root / f"{clip_id}.wav"
            if self.data_mode != 'video_only' and not audio_path.exists():
                continue 

            sequence_paths = sorted(video_sequence_root.glob(f'{clip_id}_*'))
            if self.data_mode != 'audio_only' and not sequence_paths:
                continue 

            labels = row[self.label_columns].values.astype(np.int64)

            if self.data_mode == 'audiovisual':
                for seq_path in sequence_paths:
                    samples.append({"sequence_path": seq_path, "audio_path": audio_path, "labels": labels})

            elif self.data_mode == 'audio_only':
                 if audio_path.exists():
                    samples.append({"audio_path": audio_path, "labels": labels})

        if not samples:
            raise RuntimeError(f"Found 0 valid samples for the '{self.split}' split. Check your processed data path and data_mode.")
        print(f"[{self.split}] Found {len(samples)} samples.")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        labels_tensor = torch.from_numpy(sample["labels"])

        try:
            if self.data_mode == 'audio_only':
                audio_tensor = self._load_audio(sample["audio_path"])
                return audio_tensor, labels_tensor
            
            elif self.data_mode == 'video_only':
                video_tensor = self._load_video(sample["sequence_path"])
                return video_tensor, labels_tensor
            
            else: 
                video_tensor = self._load_video(sample["sequence_path"])
                audio_tensor = self._load_audio(sample["audio_path"])
                return video_tensor, audio_tensor, labels_tensor
        except Exception as e:
            return None 

    def _load_video(self, sequence_path):
        frame_paths = sorted(sequence_path.glob("frame_*.png"))
        frames = [cv2.imread(str(p)) for p in frame_paths]
        frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames if frame is not None]
        video_tensor = torch.stack([self.video_transforms(frame) for frame in frames])
        return video_tensor.permute(1, 0, 2, 3)

    def _load_audio(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        return self.audio_transforms(waveform)

    def _get_video_transforms(self):
        return T.Compose([
            T.ToImage(), 
            T.ToDtype(torch.float32, scale=True),
            T.Resize((224, 224), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_audio_transforms(self):
        return T.Compose([
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
            torchaudio.transforms.AmplitudeToDB()
        ])


def default_collate(batch):
    """
    A default collate function that filters out None values from the batch,
    which can occur if a file is corrupt or missing.
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def create_dataloaders(config):
    """
    Factory function to create dataloaders.
    It chooses the correct Dataset class based on the config.
    """
    collate_function = default_collate
    if 'processed_data_root' in config['data']:
        print("--> Using DAiSEEProcessedDataset (for pre-processed audio/video files)")
        DatasetClass = DAiSEEProcessedDataset
    elif 'raw_data_root' in config['data']:
        print("--> Using DAiSEERawVideoDataset (for raw video files)")
        DatasetClass = DAiSEERawVideoDataset
        collate_function = collate_fn_raw_video
    else:
        raise ValueError("Config 'data' section must contain 'raw_data_root' or 'processed_data_root'")

    dataloaders = {}
    for split in ['Train', 'Validation', 'Test']:
        try:
            dataset = DatasetClass(split, config)
            if len(dataset) > 0:
                dataloaders[split] = DataLoader(
                    dataset,
                    batch_size=config['training']['batch_size'],
                    shuffle=(split == 'Train'),
                    num_workers=config['data'].get('num_workers', 2),
                    pin_memory=True,
                    collate_fn=collate_function,
                    drop_last=(split == 'Train')
                )
        except Exception as e:
            print(f"Could not create dataloader for split '{split}': {e}")
            
    return dataloaders