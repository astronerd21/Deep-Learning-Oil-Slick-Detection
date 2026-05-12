import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from src.dataset import SARDataset
from src.model import SARResNet

def parse_args():
    parser = argparse.ArgumentParser(description="Train SAR ResNet Baseline")
    parser.add_argument("--img-dir", required=True, help="Path to images_s1 folder")
    parser.add_argument("--metadata", required=True, help="Path to metadata.csv")
    parser.add_argument("--train-split", required=True, help="Path to train.txt split")
    parser.add_argument("--val-split", required=True, help="Path to val.txt split")
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2, help="Number of data loading workers")
    parser.add_argument("--output", default="checkpoints/best_model.pt")
    return parser.parse_args()

def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on: {device}")

    # Resize the raw 1024x1024 images to the required 224x224 pixels
    transform = T.Resize((224, 224), antialias=True)

    print("Loading datasets...")
    train_ds = SARDataset(args.img_dir, args.train_split, args.metadata, transform=transform)
    val_ds = SARDataset(args.img_dir, args.val_split, args.metadata, transform=transform)

    print(f"Training samples: {len(train_ds)} | Validation samples: {len(val_ds)}")

    # Use pin_memory=True for faster CPU to GPU data transfer
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=pin_mem
    )

    # Initialize model, optimizer, and loss function
    model = SARResNet(backbone=args.backbone, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    # Main training loop
    for epoch in range(1, args.epochs + 1):
        # 1. Training phase
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            # non_blocking=True allows asynchronous memory transfer to GPU
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(images)
            
        train_loss /= len(train_ds)

        # 2. Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(images)
                
                # Prediction: Logit >= 0 corresponds to Sigmoid >= 0.5 (Class 1)
                preds = (logits >= 0.0).float()
                correct += (preds == labels).sum().item()
                
        val_loss /= len(val_ds)
        val_acc = correct / len(val_ds)

        print(f"Epoch {epoch:2d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)
            print(f"  -> Saved new best model to {output_path}")

if __name__ == "__main__":
    train()