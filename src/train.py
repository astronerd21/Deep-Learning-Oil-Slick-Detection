"""Training pipeline for SAR binary oil-slick classification."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.dataset import SARDataset
from src.model import SARResNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ResNet-based binary classifier on SAR GeoTIFF data."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root directory containing 'oil_slick/' and 'no_oil_slick/'",
    )
    # Neue Argumente für die bereinigten Splits
    parser.add_argument(
        "--train-split",
        default=None,
        help="Pfad zur train_clean.txt Datei (schaltet automatischen Split ab).",
    )
    parser.add_argument(
        "--val-split-file",
        default=None,
        help="Pfad zur val_clean.txt Datei.",
    )
    parser.add_argument(
        "--backbone",
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="ResNet backbone variant (default: resnet18).",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of data reserved for validation (nur genutzt wenn keine Split-Dateien angegeben).",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from scratch instead of using ImageNet weights.",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/best.pt",
        help="Path to save the best model checkpoint.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes.",
    )
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Dataset & splits handling
    # ------------------------------------------------------------------
    if args.train_split and args.val_split_file:
        print("Lade bereinigte Splits aus Textdateien...")
        train_ds = SARDataset(root=args.data_dir, split_file=args.train_split)
        val_ds = SARDataset(root=args.data_dir, split_file=args.val_split_file)
        
        print(f"Train-Dataset: {len(train_ds)} Samples | Klassenverteilung: {train_ds.class_counts}")
        print(f"Val-Dataset: {len(val_ds)} Samples | Klassenverteilung: {val_ds.class_counts}")
        
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=device.type == "cuda"
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=device.type == "cuda"
        )
        train_size = len(train_ds)
        val_size = len(val_ds)
    else:
        # Fallback auf den alten dynamischen Random-Split
        print("Keine Split-Dateien übergeben. Führe automatischen Random-Split aus.")
        dataset = SARDataset(root=args.data_dir)
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        
        if val_size == 0:
            train_ds = dataset
            val_loader = None
        else:
            train_ds, val_ds = random_split(dataset, [train_size, val_size])
            val_loader = DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=device.type == "cuda"
            )
            
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=device.type == "cuda"
        )

    # ------------------------------------------------------------------
    # Model, optimiser, loss
    # ------------------------------------------------------------------
    model = SARResNet(backbone=args.backbone, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(images)
        train_loss /= train_size

        # --- Validate ---
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.float().to(device)
                    logits = model(images)
                    val_loss += criterion(logits, labels).item() * len(images)
                    preds = (logits.sigmoid() >= 0.5).long()
                    correct += (preds == labels.long()).sum().item()
            val_loss /= val_size
            val_acc = correct / val_size

            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), output_path)
                print(f"  → Saved best model to '{output_path}'")
        else:
            print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.4f}")


if __name__ == "__main__":
    train()