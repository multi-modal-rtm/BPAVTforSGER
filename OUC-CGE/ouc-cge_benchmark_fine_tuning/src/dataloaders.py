import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchvision.transforms import v2 as T
from decord import VideoReader, cpu
import numpy as np
import librosa 

def collate_fn(batch):
    """
    Custom collate function that filters out None values.
    This prevents the dataloader from crashing if a file is corrupt.
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


class OUC_CGEDataset(Dataset):
    """
    Custom PyTorch Dataset for the OUC-CGE dataset.
    Loads video frames and audio directly from MP4 files on-the-fly.
    
    This dataset is "split-aware" and applies augmentations
    only to the 'train' split.
    """
    def __init__(self, csv_path, root_dir, config, split='train'): 
        
        print("\n--- LOADING DATALOADER ---")
        print(f"CSV Path: {csv_path}")
        print(f"Split: {split}")
        print("--------------------------")
        
        try:
            self.annotations = pd.read_csv(csv_path, header=None, sep=' ')
            self.annotations.columns = ['video_path', 'label'] 
        except Exception as e:
            print(f"FATAL: Could not read CSV file at {csv_path}. Error: {e}")
            raise e
            
        self.root_dir = root_dir
        self.config = config
        self.data_mode = config['data_mode']
        self.split = split 

        self.train_transform = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize(size=(256, 256), antialias=True), 
            T.RandomCrop(224),
            T.RandomHorizontalFlip(p=0.5),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.val_test_transform = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize(size=(224, 224), antialias=True),
            T.CenterCrop(224), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        n_mels = config.get('n_mels', 128)
        print(f"Initializing dataloader with n_mels: {n_mels}")

        self.audio_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=160,
            n_mels=n_mels
        )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        video_relative_path = self.annotations.iloc[index, 0] 
        try:
            label = int(self.annotations.iloc[index, 1]) 

            if video_relative_path.startswith('videos/') or video_relative_path.startswith('videos\\'):
                video_relative_path = video_relative_path[7:]
            
            video_path = os.path.join(self.root_dir, video_relative_path)
            video_path = os.path.normpath(video_path)

            video_tensor = torch.zeros((self.config['num_frames'], 3, 224, 224))
            audio_tensor = torch.zeros((128, self.config['target_audio_len']))

            if self.data_mode in ['video_only', 'audiovisual']:
                vr = VideoReader(video_path, ctx=cpu(0))
                if len(vr) == 0: raise RuntimeError("Video has 0 frames")
                
                total_frames = len(vr)
                frame_indices = np.linspace(0, total_frames - 1, self.config['num_frames'], dtype=int)
                frames = vr.get_batch(frame_indices).asnumpy()

                frames_list = [frame for frame in frames] 

                if self.split == 'train':
                    video_tensor_list = self.train_transform(frames_list)
                    video_tensor = torch.stack(video_tensor_list, dim=0) 
                else:

                    video_tensor_list = self.val_test_transform(frames_list)
                    video_tensor = torch.stack(video_tensor_list, dim=0)

            if self.data_mode in ['audio_only', 'audiovisual']:
                waveform, sample_rate = librosa.load(video_path, sr=16000, mono=True)
                if waveform.size == 0: raise RuntimeError("Audio stream is empty")

                waveform = torch.from_numpy(waveform).unsqueeze(0)
                
                spectrogram = self.audio_transform(waveform)

                if spectrogram.shape[2] > self.config['target_audio_len']:
                    spectrogram = spectrogram[:, :, :self.config['target_audio_len']]
                else:
                    spectrogram = torch.nn.functional.pad(spectrogram, (0, self.config['target_audio_len'] - spectrogram.shape[2]))
                
                audio_tensor = spectrogram.squeeze(0)

            return {"video": video_tensor, "audio": audio_tensor, "label": torch.tensor(label, dtype=torch.long)}

        except Exception as e:
            return None

def create_dataloaders(config):
    """Creates training, validation, and test dataloaders."""
    train_csv_path = os.path.join(config['root_dir'], config['train_csv'])
    val_csv_path = os.path.join(config['root_dir'], config['val_csv'])
    test_csv_path = os.path.join(config['root_dir'], config['test_csv'])

    train_dataset = OUC_CGEDataset(
        csv_path=train_csv_path, 
        root_dir=config['root_dir'], 
        config=config, 
        split='train'
    )
    val_dataset = OUC_CGEDataset(
        csv_path=val_csv_path, 
        root_dir=config['root_dir'], 
        config=config, 
        split='val'  
    )
    test_dataset = OUC_CGEDataset(
        csv_path=test_csv_path, 
        root_dir=config['root_dir'], 
        config=config, 
        split='test' 
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader