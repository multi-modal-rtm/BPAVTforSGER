import torch
import torchaudio
import torchvision.transforms as T
from torchvision.io import read_video
import torch.nn.functional as F
import numpy as np

# Configuration for different models' I/O requirements
MODEL_IO = {
    "mbt": dict(n_frames=96, fps=25, size=224,
                audio_sr=16000, span_s=8, audio_feat="mel"),
    "vatt": dict(n_frames=16, fps=30, size=224,
                  audio_sr=48000, span_s=2, audio_feat="wave"),
    "avt": dict(n_frames=32, fps=30, size=224,
                 audio_sr=16000, span_s=4, audio_feat="mel"),
    "swin": dict(n_frames=32, fps=15, size=224,
                  audio_sr=None,   span_s=None, audio_feat=None),
}

def build_video_transform(model_name: str) -> T.Compose:
    """
    Create spatial transforms for video frames based on model I/O specs.
    Returns a torchvision.transforms.Compose that can be applied per-frame.
    """
    io = MODEL_IO[model_name]
    return T.Compose([
        T.Resize(io["size"] + 32),
        T.RandomCrop(io["size"]),
        T.RandomHorizontalFlip(),
        T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    ])


def extract_video(video_path: str, model_name: str) -> torch.Tensor:
    """
    Load a video from disk, resample to target FPS, sample or pad to n_frames,
    and return a tensor of shape [C, T, H, W] normalized to [0,1].
    """
    io = MODEL_IO[model_name]
    # Read all frames (T, H, W, C)
    video, _, info = read_video(video_path, pts_unit="sec")
    orig_fps = float(info.get('video_fps', io['fps']))

    # Resample frames to model's FPS if different
    if orig_fps != io["fps"]:
        num_frames = video.shape[0]
        duration = num_frames / orig_fps
        new_num = int(duration * io["fps"])
        times = torch.linspace(0, duration, steps=new_num)
        indices = (times * orig_fps).long().clamp(max=num_frames - 1)
        video = video[indices]

    # Uniformly sample or pad to n_frames
    T_total = video.shape[0]
    if T_total >= io["n_frames"]:
        idxs = np.linspace(0, T_total - 1, io["n_frames"], dtype=int)
        video = video[idxs]
    else:
        pad_amt = io["n_frames"] - T_total
        last = video[-1:].expand(pad_amt, -1, -1, -1)
        video = torch.cat([video, last], dim=0)

    # Permute to [C, T, H, W] and normalize
    video = video.permute(3, 0, 1, 2).float() / 255.0
    return video


def extract_audio(audio_path: str, model_name: str) -> torch.Tensor:
    """
    Load audio, resample if needed, and return either raw waveform or log-mel spectrogram.
    Returns None if audio_feat is None.
    """
    io = MODEL_IO[model_name]
    if io["audio_feat"] is None:
        return None

    waveform, sr = torchaudio.load(audio_path)  # [channels, time]
    # Convert to mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to target sample rate
    if sr != io["audio_sr"]:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=io["audio_sr"])
        waveform = resampler(waveform)

    # Return raw waveform clipped/padded
    if io["audio_feat"] == "wave":
        expected_len = int(io["audio_sr"] * io["span_s"])
        if waveform.size(1) >= expected_len:
            return waveform[:, :expected_len]
        else:
            pad_len = expected_len - waveform.size(1)
            return F.pad(waveform, (0, pad_len))

    # Compute log-mel spectrogram
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=io["audio_sr"], n_mels=64
    )(waveform)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    return mel_db  # [1, n_mels, time]
