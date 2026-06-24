"""Evaluation script to compute Accuracy, F1 Score, AUROC, and Confusion Matrix."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from src.dataset import SARDataset
from src.model import SARResNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SARResNet model on a given dataset split."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root directory containing SAR image files.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the saved model checkpoint (.pt).",
    )
    parser.add_argument(
        "--split",
        required=True,
        help="Path to the split text file (e.g., splits/random/test.txt).",
    )
    parser.add_argument(
        "--backbone",
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="ResNet backbone variant (default: resnet18).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    transform = T.Compose(
        [
            T.Resize((224, 224), antialias=True),
        ]
    )
    split_path = Path(args.split)
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file missing at '{split_path}'")

    print(f"Loading split '{split_path.name}'...")
    dataset = SARDataset(
        root=args.data_dir, split_file=str(split_path), transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SARResNet(backbone=args.backbone, pretrained=False).to(device)

    print(f"Loading checkpoint '{args.model}'...")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("  → Loaded weights from dictionary checkpoint.")
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = logits.sigmoid().cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = np.mean(all_preds == all_targets)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    auroc = roc_auc_score(all_targets, all_probs)
    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel()

    split_name = split_path.stem
    print("\n" + "=" * 55)
    print(f"=== EVALUATION REPORT: {split_name.upper()} ===")
    print("=" * 55)
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUROC:     {auroc:.4f}\n")
    print("Confusion Matrix:")
    print("                 Pred: NO OIL    Pred: OIL")
    print(f"Actual: NO OIL | TN: {tn:<9} | FP: {fp:<9} |")
    print(f"Actual: OIL    | FN: {fn:<9} | TP: {tp:<9} |")
    print("=" * 55)

    out_dir = Path("evaluation_outputs")
    out_dir.mkdir(exist_ok=True)
    fp_file = out_dir / f"false_positives_{split_name}.txt"

    sample_items = dataset.samples
    with open(fp_file, "w") as f:
        for idx, (p, t) in enumerate(zip(all_preds, all_targets)):
            if p == 1 and t == 0:
                f.write(f"{sample_items[idx][0].name}\n")

    print(f"Exported False Positive filenames (Lookalikes) to: '{fp_file}'")


if __name__ == "__main__":
    main()
