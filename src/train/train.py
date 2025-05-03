import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train(model, dataset, device='cuda', epochs=10, lr=1e-4, batch_size=4, model_name="model"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['labels'].to(device).long()

            outputs = model(video, audio)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

        # Save every epoch
        torch.save(model.state_dict(), f"checkpoints/{model_name}_epoch{epoch+1}.pt")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"checkpoints/{model_name}_best.pt")
