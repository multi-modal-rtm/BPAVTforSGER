import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm

def evaluate(model, dataset, device='cuda', batch_size=4):
    loader = DataLoader(dataset, batch_size=batch_size)
    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            if 'video' in batch:
                video = batch['video'].to(device)
            if 'audio' in batch:
                audio = batch['audio'].to(device)
            if 'labels' in batch:
                labels = batch['labels'].to(device).long()

            outputs = model(video, audio)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    ap = average_precision_score(all_labels, all_probs)
    ccc = pearsonr(all_labels, all_probs)[0]

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"mAP: {ap:.4f}")
    print(f"CCC: {ccc:.4f}")
    return {"accuracy": acc, "f1": f1, "auc": auc, "map": ap, "ccc": ccc}