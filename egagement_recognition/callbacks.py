import os
import torch

class EarlyStopping:
    """
    Stops training if validation accuracy doesn’t improve for `patience` epochs.
    """
    def __init__(self, patience: int = 5, delta: float = 0.0, verbose: bool = True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_score: float):
        if self.best_score is None or val_score > self.best_score + self.delta:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

class LRSchedulerCallback:
    """
    Wraps a torch.optim.lr_scheduler to step it on epoch end (or batch).
    """
    def __init__(self, scheduler, step_on_batch: bool = False):
        """
        scheduler: a torch.optim.lr_scheduler instance
        step_on_batch: if True, call scheduler.step() every batch; otherwise every epoch
        """
        self.scheduler = scheduler
        self.step_on_batch = step_on_batch

    def on_batch_end(self):
        if self.step_on_batch:
            self.scheduler.step()

    def on_epoch_end(self):
        if not self.step_on_batch:
            self.scheduler.step()

class CheckpointCallback:
    """
    Additional checkpointing logic beyond save_checkpoint, e.g. keep last N or save every M epochs.
    """
    def __init__(self, ckpt_dir: str, save_every: int = 1, keep_last_n: int = 3):
        self.ckpt_dir = ckpt_dir
        self.save_every = save_every
        self.keep_last_n = keep_last_n

    def __call__(self, state: dict, epoch: int):
        if (epoch + 1) % self.save_every == 0:
            path = os.path.join(self.ckpt_dir, f"epoch_{epoch+1}.pth")
            torch.save(state, path)
            # prune old
            all_ckpts = sorted([f for f in os.listdir(self.ckpt_dir) if f.startswith("epoch_")])
            if len(all_ckpts) > self.keep_last_n:
                to_remove = all_ckpts[:-self.keep_last_n]
                for fn in to_remove:
                    os.remove(os.path.join(self.ckpt_dir, fn))
