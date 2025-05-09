
# BPAVTforSGER: Benchmarking Pre-trained Audio-Visual Transformers for Small-Group Engagement Recognition

This repository implements a unified pipeline for benchmarking A/V transformers on three engagement datasets (DAiSEE, OUC-CGE, AMI). It supports four models (MBT, VATT, AVT, Video-Swin) with pretrained Kinetics-400 weights when available.

---

## 📁 Project Structure


engagement\_recognition/
├── configs/
│   └── default.yaml
├── data/
│   ├── manifests/           # CSV manifests for each dataset
│   └── raw/                 # raw video/audio files
├── engagement\_recognition/  # core package
│   ├── **init**.py
│   ├── preprocess.py        # video/audio featurizers & transforms
│   ├── dataloader.py        # Dataset & DataLoader factories
│   ├── models.py            # model definitions & weight loading
│   ├── train.py             # training loop & checkpointing
│   ├── test.py              # evaluation, metrics, checkpoint loading
│   ├── utils.py             # checkpointing, meters, mAP
│   └── callbacks.py         # (optional) schedulers / early stopping
├── scripts/
│   ├── download\_data.sh     # helper to download raw datasets
│   └── extract\_features.py  # offline feature extraction
├── outputs/
│   ├── logs/                # TensorBoard logs
│   └── checkpoints/         # model checkpoints
├── requirements.txt
└── README.md

````

---

## 📦 Installation

```bash
# Clone repo
git clone https://github.com/yourusername/engagement_recognition.git
cd engagement_recognition

# Create env & install
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
````

**`requirements.txt`** should include, at minimum:

```
torch
torchvision
torchaudio
omegaconf
numpy
pandas
scikit-learn
scenic       # for MBT
huggingface-hub
swin-transformer
# plus your VATT/AVT repos
```

---

## 📥 Download & Prepare Data

Edit `scripts/download_data.sh` with correct URLs and run:

```bash
bash scripts/download_data.sh
```

This should:

1. Download and unzip **DAiSEE**, **OUC-CGE**, and **AMI** into `data/raw/`.
2. Generate CSV manifests in `data/manifests/` containing:

   * `video_path`
   * `audio_path`
   * `label` (mapped 0–3)
   * `split` (train/val/test)

---

## ⚙️ Configuration

All hyperparameters and paths live in `configs/default.yaml`.
Adjust:

* `data.roots` → where your raw folders live
* `model.name` → one of `mbt,vatt,avt,swin`
* `training.epochs, lr, fp16, resume_from`
* `logging.log_dir, ckpt_dir, log_interval`

---

## 🚀 Training

```bash
# From repo root
python -m engagement_recognition.train
```

* Checkpoints (`last.pth`, `best.pth`) appear under `outputs/checkpoints/`.
* TensorBoard logs under `outputs/logs/` → `tensorboard --logdir outputs/logs`

---

## 🔍 Testing & Evaluation

```bash
python -m engagement_recognition.test
```

This prints:

* Classification Report
* Confusion Matrix
* **mAP** and per-class AP

---

## 🗄 Offline Feature Extraction

If you’d like to precompute and cache embeddings:

```bash
python scripts/extract_features.py \
  --model mbt \
  --datasets daisee ouc_cge ami \
  --out_dir data/features
```

This script should:

1. Load each video/audio clip.
2. Run through `preprocess.extract_video` / `extract_audio`.
3. Forward through frozen backbone.
4. Save per-clip `.npy` feature files to `data/features/{dataset}/{model}/`.

---

## 🔧 Callbacks & Extensions

* **callbacks.py** is a placeholder for learning-rate schedulers, early stopping, etc.
* Feel free to integrate PyTorch Lightning or Hydra for more flexibility.

---

## 📖 Citation

If you use this code in publications, please cite:

> “Benchmarking Pre-trained Audio-Visual Transformers for Small-Group Engagement Recognition,” *BMVC 2025*.


