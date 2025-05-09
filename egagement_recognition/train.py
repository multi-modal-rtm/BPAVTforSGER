import os
import torch
import yaml
from omegaconf import OmegaConf
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from engagement_recognition.dataloader import make_dataloader
from engagement_recognition.models import EngagementNet
from engagement_recognition.utils import save_checkpoint, AverageMeter, load_checkpoint

def train():
    # ─── Load config ──────────────────────────────
    cfg = OmegaConf.load(os.path.join("configs", "default.yaml"))

    # ─── Prepare logging ──────────────────────────
    os.makedirs(cfg.logging.log_dir, exist_ok=True)
    os.makedirs(cfg.logging.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=cfg.logging.log_dir)

    # ─── Data loaders ─────────────────────────────
    train_loader = make_dataloader("train", cfg.model.name)
    val_loader   = make_dataloader("val",   cfg.model.name)

    # ─── Model, optimizer, loss ───────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EngagementNet(cfg.model.name, cfg.model.num_classes).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay
    )
    criterion = CrossEntropyLoss()
    scaler = GradScaler(enabled=cfg.training.fp16)

    # ─── Optionally resume ────────────────────────
    start_epoch = 0
    best_acc = 0.0
    if cfg.training.get("resume_from"):
        ckpt = load_checkpoint(cfg.training.resume_from, model, optimizer, scaler)
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt["best_acc"]

    # ─── Training loop ────────────────────────────
    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()
        loss_meter = AverageMeter()
        for i, batch in enumerate(train_loader):
            vids, auds, labels = batch["video"], batch["audio"], batch["label"]
            vids = vids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if auds is not None:
                auds = auds.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(enabled=cfg.training.fp16):
                outputs = model(vids, auds)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), vids.size(0))
            if i % cfg.logging.get("log_interval", 50) == 0:
                step = epoch * len(train_loader) + i
                writer.add_scalar("train/loss", loss_meter.avg, step)

        # ─── Validation ────────────────────────────
        val_acc = validate(model, val_loader, device)
        writer.add_scalar("val/accuracy", val_acc, epoch)

        print(f"[Epoch {epoch:02d}] Train Loss: {loss_meter.avg:.4f}  Val Acc: {val_acc:.4f}")

        # ─── Checkpoint ────────────────────────────
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
            "epoch":     epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_acc":  best_acc
        }, is_best, cfg.logging.ckpt_dir)

    writer.close()

def validate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            vids, auds, labels = batch["video"], batch["audio"], batch["label"]
            vids = vids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if auds is not None:
                auds = auds.to(device, non_blocking=True)

            logits = model(vids, auds)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

if __name__ == "__main__":
    train()
