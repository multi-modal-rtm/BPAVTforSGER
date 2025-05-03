# BPAVTforSGER: Benchmarking Pre-trained Audio-Visual Transformers for Small-Group Engagement Recognition

This repository benchmarks pre-trained and scratch-built audio-visual transformer models on small-group engagement datasets like **OUC-CGE**, **AMI**, and **CMOSE**.

---

## 📁 Project Structure

```
project-root/
├── src/
│   ├── models/               # All model architectures (scratch and pretrained)
│   ├── data/                 # Dataloader and preprocessing scripts
│   ├── train/                # Training and evaluation modules
│   └── config.yaml           # Model, dataset, and training configuration
├── data/                     # Raw data and pre-extracted features (not tracked)
├── checkpoints/              # Saved model checkpoints (not tracked)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Supported Models

You can switch models by setting the `model:` field in `config.yaml`:

### 🛠️ Scratch Models (shared backbone: ResNet + Wav2Vec2)
- `MBTScratch`
- `VATTScratch`
- `AVAdapterScratch`
- `CrossModalAdapterScratch`

### 🧠 Pretrained Models (ViT + Wav2Vec2)
- `VATTPretrained`
- `AVAdapterPretrained`
- `CrossModalAdapter`

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Organize your data under `data/OUC-CGE/` or similar.

3. Modify `src/config.yaml` to select your model and dataset path:
   ```yaml
   model: "AVAdapterPretrained"
   ```

4. Run training + evaluation:
   ```bash
   python src/train/launch.py
   ```

---

## 📊 Evaluation Metrics
This framework supports:
- Accuracy / Balanced Accuracy
- F1 Score
- AUC-ROC
- mAP (mean Average Precision)
- CCC (Concordance Correlation Coefficient)
