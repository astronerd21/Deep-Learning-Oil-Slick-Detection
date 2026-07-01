"""Training pipeline for SAR binary oil-slick classification."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T

from src.dataset import SARDataset
from src.model import SARResNet, TerraMindClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ResNet-based binary classifier on SAR GeoTIFF data."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root directory containing SAR image files.",
    )
    parser.add_argument(
        "--train-split",
        default=None,
        help="Path to the train_clean.txt file.",
    )
    parser.add_argument(
        "--val-split-file",
        default=None,
        help="Path to the val_clean.txt file.",
    )
    parser.add_argument(
        "--backbone",
        default="resnet18",
        choices=["resnet18", "resnet50", "terramind"],
        help="ResNet backbone variant (default: resnet18).",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of data reserved for validation (only used if no split files are provided).",
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

    train_transform = T.Compose(
        [
            T.Resize((224, 224), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ]
    )

    val_transform = T.Compose(
        [
            T.Resize((224, 224), antialias=True),
        ]
    )

    if args.train_split and args.val_split_file:
        print("Loading cleaned splits from text files...")
        train_ds = SARDataset(
            root=args.data_dir, split_file=args.train_split, transform=train_transform
        )
        val_ds = SARDataset(
            root=args.data_dir, split_file=args.val_split_file, transform=val_transform
        )

        print(
            f"Train-Dataset: {len(train_ds)} samples | Class distribution: {train_ds.class_counts}"
        )
        print(
            f"Val-Dataset: {len(val_ds)} samples | Class distribution: {val_ds.class_counts}"
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        train_size = len(train_ds)
        val_size = len(val_ds)
    else:
        print("No split files provided. Running automatic random split fallback.")
        base_dataset = SARDataset(root=args.data_dir, split_file=None, transform=None)

        val_size = int(len(base_dataset) * args.val_split)
        train_size = len(base_dataset) - val_size

        if val_size == 0:
            train_ds = SARDataset(
                root=args.data_dir, split_file=None, transform=train_transform
            )
            val_loader = None
        else:
            train_indices, val_indices = random_split(
                base_dataset, [train_size, val_size]
            )

            full_train_ds = SARDataset(
                root=args.data_dir, split_file=None, transform=train_transform
            )
            full_val_ds = SARDataset(
                root=args.data_dir, split_file=None, transform=val_transform
            )

            from torch.utils.data import Subset

            train_ds = Subset(full_train_ds, train_indices.indices)
            val_ds = Subset(full_val_ds, val_indices.indices)

            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    if args.backbone in ["resnet18", "resnet50"]:
        model = SARResNet(backbone=args.backbone, pretrained=not args.no_pretrained).to(
            device
        )
    elif args.backbone == "terramind":
        model = TerraMindClassifier(freeze_backbone=True).to(device)
    else:
        raise ValueError(f"Unknown backbone {args.backbone}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    criterion = nn.BCEWithLogitsLoss()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = output_path.parent / "training_log.csv"

    best_val_loss = float("inf")

    if not csv_path.exists():
        with open(csv_path, "w") as f:
            f.write("epoch,train_loss,val_loss,val_acc\n")

    if output_path.exists():
        print(
            f"Found existing checkpoint at '{output_path}'. Checking for historical best validation loss..."
        )
        try:
            checkpoint = torch.load(
                output_path, map_location=device, weights_only=False
            )
            if isinstance(checkpoint, dict) and "best_val_loss" in checkpoint:
                best_val_loss = checkpoint["best_val_loss"]
                print(
                    f"  → Resuming with historical best_val_loss: {best_val_loss:.4f}"
                )
            else:
                print(
                    "  → Checkpoint is in old format (missing 'best_val_loss' key). Starting tracker from infinity."
                )
        except Exception as e:
            print(f"  → Warning: Could not load existing checkpoint to check loss: {e}")

    for epoch in range(1, args.epochs + 1):
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

            with open(csv_path, "a") as f:
                f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{val_acc:.4f}\n")

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                }
                torch.save(checkpoint, output_path)
                print(f"  → Saved new best model to '{output_path}'")


if __name__ == "__main__":
    train()
