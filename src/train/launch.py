import yaml
import torch
from models import *
from train.train import train
from train.evaluate import evaluate
from data.dataloader import OUCGEVideoDataset

def main(config_path="src/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    dataset = OUCGEVideoDataset(
        video_dir=cfg["dataset"]["video_dir"],
        label_dir=cfg["dataset"]["label_dir"],
        fps=1,
        duration=10
    )

    train_dataset = dataset
    val_dataset = dataset

    model_class = globals()[cfg["model"]]
    model = model_class()

    train(model, train_dataset,
          device=cfg["training"]["device"],
          epochs=cfg["training"]["epochs"],
          lr=cfg["training"]["learning_rate"],
          batch_size=cfg["training"]["batch_size"])

    evaluate(model, val_dataset,
             device=cfg["training"]["device"],
             batch_size=cfg["training"]["batch_size"])

if __name__ == "__main__":
    main()