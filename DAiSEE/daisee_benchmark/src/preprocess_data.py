import os
import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from tqdm import tqdm


RAW_DATASET_ROOT = Path('D:/Abdulaziz/DAiSEE_Dataset_Raw/daisee_dataset/DAiSEE')
PROCESSED_DATA_ROOT = Path('D:/Abdulaziz/daisee_benchmark/data/processed')

FRAME_SIZE = 224
FPS = 15
SEQUENCE_LENGTH = 16
AUDIO_SAMPLE_RATE = 16000

def extract_frames_and_audio(video_path):
    """
    Extracts frames and audio from a single video file.
    This version is more resilient to audio extraction errors.
    """
    frames = []
    audio = None
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Warning: could not open video {video_path}")
            return None, None

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0: return None, None
            
        frame_interval = max(1, int(video_fps / FPS))
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_count % frame_interval == 0:
                frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            frame_count += 1
        cap.release()
    except Exception as e:
        print(f"Warning: Failed to extract frames from {video_path}. Error: {e}")
        return None, None 

    try:
        audio, _ = librosa.load(str(video_path), sr=AUDIO_SAMPLE_RATE, duration=10)
    except Exception as e:
        # This will now print the specific error librosa is having
        print(f"\n---> Librosa Error on file {video_path.name}: {e}\n")
        audio = None
    
    # try:
    #     audio, _ = librosa.load(str(video_path), sr=AUDIO_SAMPLE_RATE, duration=10)
    # except Exception:
    #     audio = None

    return np.array(frames), audio

def create_and_save_sequences(frames, audio, clip_id, split):
    """Saves the extracted frames and audio."""
    num_frames = len(frames)
    if num_frames < SEQUENCE_LENGTH: return

    for i in range(0, num_frames - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH // 2):
        frame_sequence = frames[i:i + SEQUENCE_LENGTH]
        if len(frame_sequence) == SEQUENCE_LENGTH:
            sequence_dir = PROCESSED_DATA_ROOT / split / 'video' / f"{clip_id}_{i}"
            sequence_dir.mkdir(parents=True, exist_ok=True)
            for j, frame in enumerate(frame_sequence):
                frame_path = sequence_dir / f"frame_{j:04d}.png"
                cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    if audio is not None:
        audio_dir = PROCESSED_DATA_ROOT / split / 'audio'
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / f"{clip_id}.wav"
        sf.write(str(audio_path), audio, AUDIO_SAMPLE_RATE)

def main():
    """Main function to run preprocessing."""
    print("--- Starting Preprocessing ---")
    print(f"Reading raw data from: {RAW_DATASET_ROOT}")
    print(f"Saving processed data to: {PROCESSED_DATA_ROOT}")

    for split in ['Train', 'Test', 'Validation']:
        print(f"\n--- Processing '{split}' data ---")
        split_video_dir = RAW_DATASET_ROOT / 'DataSet' / split
        if not split_video_dir.exists():
            print(f"WARNING: Directory does not exist: {split_video_dir}. Skipping.")
            continue

        video_paths = list(split_video_dir.rglob('*.avi'))
        video_paths.extend(list(split_video_dir.rglob('*.mp4')))
        print(f"Found {len(video_paths)} videos in '{split}' set.")

        if not video_paths: continue

        for video_path in tqdm(video_paths, desc=f"Processing {split} videos"):
            clip_id = video_path.stem
            frames, audio = extract_frames_and_audio(video_path)
            if frames is not None and len(frames) > 0:
                create_and_save_sequences(frames, audio, clip_id, split)

    print("\n--- Preprocessing complete! ---")

if __name__ == "__main__":
    main()