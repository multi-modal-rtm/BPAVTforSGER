import os
import torch
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import classification_report, confusion_matrix

from engagement_recognition.dataloader import make_dataloader
from engagement_recognition.models import EngagementNet
from engagement_recognition.utils import compute_mAP

def test():
    # ─── Load config ─────────────────────────────────────────
    cfg = OmegaConf.load(os.path.join("configs", "default.yaml"))

    # ─── Prepare model ───────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EngagementNet(cfg.model.name, cfg.model.num_classes).to(device)

    # ─── Load best checkpoint ─────────────────────────────────
    ckpt_path = os.path.join(cfg.logging.ckpt_dir, "best.pth")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # ─── Data loader for test split ───────────────────────────
    test_loader = make_dataloader("test", cfg.model.name)

    # ─── Run inference ────────────────────────────────────────
    all_logits = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            vids = batch["video"].to(device, non_blocking=True)
            auds = (batch["audio"].to(device, non_blocking=True)
                    if batch["audio"] is not None else None)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(vids, auds)  # [B, C]
            probs  = torch.softmax(logits, dim=1)

            all_logits.append(probs.cpu().numpy())
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # ─── Stack results ────────────────────────────────────────
    all_logits = np.vstack(all_logits)  # [N, C]
    all_labels = np.array(all_labels)   # [N]

    # ─── Classification Report & Confusion Matrix ────────────
    target_names = ["disengaged", "neutral", "engaged", "high-engaged"]
    print("Classification Report:\n")
    print(classification_report(all_labels, all_preds,
                                target_names=target_names,
                                zero_division=0))
    print("Confusion Matrix:\n")
    print(confusion_matrix(all_labels, all_preds))

    # ─── Compute mAP ──────────────────────────────────────────
    mAP, ap_per_class = compute_mAP(all_labels, all_logits, cfg.model.num_classes)
    print(f"\nmAP: {mAP:.4f}")
    for idx, ap in enumerate(ap_per_class):
        print(f"  AP class {target_names[idx]}: {ap:.4f}")

if __name__ == "__main__":
    test()
