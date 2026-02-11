import torch
import pandas as pd
import numpy as np
import librosa
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToFloatAndScale:
    """A callable class to convert a uint8 tensor to a float32 tensor and scale it to [0, 1]."""
    def __call__(self, tensor):
        return tensor.to(torch.float32) / 255.0

def collate_fn(batch):
    """Filters out None values (samples that failed to load) from a batch."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def read_and_process_labels(file_path):
    """Reads CSV or Excel, handles headers, filters irrelevant labels, and maps strings to integers."""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, header=0)
        else:
            raise ValueError("Unsupported label file format. Please use .csv or .xlsx")
    except FileNotFoundError:
        logger.error(f"Label file not found at: {file_path}")
        return None

    df.columns = [str(col).strip() for col in df.columns]
    
    if df.shape[1] >= 3:
        if not hasattr(read_and_process_labels, "warned"):
            logger.info(f"Dataloader: Detected {df.shape[1]} columns. Assuming Video ID is in column 2 and Label is in column 3.")
            read_and_process_labels.warned = True
        df = df.iloc[:, [1, 2]]
    else:
        df = df.iloc[:, [0, 1]]
        
    df.columns = ['video_id', 'engagement_level_str']

    irrelevant_labels = ['SNP(Subject Not Present)', 'label']
    df = df[~df['engagement_level_str'].isin(irrelevant_labels)]

    unique_labels = sorted(df['engagement_level_str'].unique())
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    if not hasattr(read_and_process_labels, "logged"):
        logger.info("--- Dataloader: Mapping String Labels to Integers ---")
        for label, index in label_map.items():
            logger.info(f"'{label}' -> {index}")
        read_and_process_labels.logged = True

    df['engagement_level'] = df['engagement_level_str'].map(label_map)
    return df

class EngageNetDataset(Dataset):
    def __init__(self, annotation_file, config, split, transform=None, audio_transform=None):
        self.annotations = read_and_process_labels(annotation_file)
        if self.annotations is None:
            raise FileNotFoundError(f"Could not load annotations from {annotation_file}")
        self.config = config
        self.split = split 
        self.transform = transform
        self.audio_transform = audio_transform
        self.modality = config.modality

    def __len__(self):
        return len(self.annotations)

    def _sample_frames(self, vr, num_frames_to_sample):
        total_frames = len(vr)
        if total_frames < num_frames_to_sample:
            indices = np.random.choice(total_frames, num_frames_to_sample, replace=True)
        else:
            indices = np.linspace(0, total_frames - 1, num=num_frames_to_sample, dtype=int)
        indices.sort()
        return indices

    def _load_video(self, video_path):
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            frame_indices = self._sample_frames(vr, self.config.num_frames)
            frames = vr.get_batch(frame_indices).asnumpy()
            return torch.from_numpy(frames).permute(0, 3, 1, 2)
        except Exception:
            return None

    def _load_audio(self, video_path):
        try:
            y, sr = librosa.load(video_path, sr=self.config.audio_sampling_rate, mono=True)
            target_len = self.config.input_clip_length * sr
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)), 'constant')
            else:
                y = y[:target_len]
            return torch.from_numpy(y).float()
        except Exception:
            return None

    def __getitem__(self, index):
        video_filename = self.annotations.iloc[index]['video_id']
        video_id = video_filename.split('.')[0]
        
        split_folder_map = {
            'train': 'Train',
            'val': 'Validation',
            'test': 'Test'
        }
        split_folder_name = split_folder_map.get(self.split, self.split.capitalize())
        video_path = os.path.join(self.config.data_root, split_folder_name, f"{video_id}.mp4")
        
        label = int(self.annotations.iloc[index]['engagement_level'])
        data = {'label': label}

        if self.modality == 'video':
            video_frames = self._load_video(video_path)
            if video_frames is None: return None
            if self.transform: video_frames = self.transform(video_frames)
            data['video'] = video_frames
        
        elif self.modality == 'audio':
            audio_clip = self._load_audio(video_path)
            if audio_clip is None: return None
            if self.audio_transform: audio_clip = self.audio_transform(audio_clip)
            data['audio'] = audio_clip

        elif self.modality == 'audio_visual':
            video_frames = self._load_video(video_path)
            audio_clip = self._load_audio(video_path)
            if video_frames is None or audio_clip is None: return None
            if self.transform: video_frames = self.transform(video_frames)
            if self.audio_transform: audio_clip = self.audio_transform(audio_clip)
            data['video'] = video_frames
            data['audio'] = audio_clip
            
        return data

def get_transforms(config, split='train'):
 
    video_transform_list = [
        ToFloatAndScale(),
        transforms.Resize((224, 224), antialias=True),
    ]


    if split == 'train':
        video_transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        ])

    video_transform_list.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    class MelSpectrogram(object):
        def __init__(self, params):
            self.params = params
        def __call__(self, waveform):
            spec = librosa.feature.melspectrogram(
                y=waveform.numpy(), sr=self.params.audio_sampling_rate,
                n_fft=self.params.audio_params.n_fft,
                hop_length=self.params.audio_params.hop_length,
                n_mels=self.params.audio_params.n_mels
            )
            spec_db = librosa.power_to_db(spec, ref=np.max)
            return torch.from_numpy(spec_db).float()

    audio_transform = MelSpectrogram(config) if hasattr(config, 'audio_params') else None

    return {
        'video': transforms.Compose(video_transform_list),
        'audio': audio_transform
    }

def create_dataloader(config, split='train'):
    annotation_path = getattr(config, f"{split}_csv")
    transforms_dict = get_transforms(config, split)
    
    dataset = EngageNetDataset(
        annotation_file=annotation_path, config=config, split=split,
        transform=transforms_dict.get('video'),
        audio_transform=transforms_dict.get('audio')
    )
    
    dataloader_kwargs = {
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'shuffle': (split == 'train'),
        'pin_memory': True,
        'drop_last': (split == 'train'),
        'collate_fn': collate_fn
    }
    if config.num_workers > 0:
        dataloader_kwargs['persistent_workers'] = True
        logger.info("Using persistent workers for DataLoader.")

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    
    logger.info(f"Created {split} dataloader with {len(dataset)} samples.")
    return dataloader