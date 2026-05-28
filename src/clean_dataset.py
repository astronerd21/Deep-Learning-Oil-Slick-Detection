"""Script to clean SAR image chips by filtering out high NoData content."""

import argparse
from pathlib import Path
import rasterio
import numpy as np


def check_nodata_ratio(filepath: Path, threshold: float = 0.5) -> bool:
    """Returns True if the VV band contains more 0.0 pixels than the threshold."""
    with rasterio.open(filepath) as src:
        vv_band = src.read(1)
    nodata_pixels = np.sum(vv_band == 0.0)
    total_pixels = vv_band.size
    return (nodata_pixels / total_pixels) > threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter out SAR chips with high NoData edge artifacts.")
    parser.add_argument("--data-dir", required=True, help="Root-Pfad zum Datensatz-Ordner.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Anteil der Daten für Validierung.")
    args = parser.parse_args()

    root_path = Path(args.data_dir)
    
    # Stratifiziertes Sammeln anhand eurer Namenskonvention
    all_valid_by_class = {0: [], 1: []}
    
    # Direkt alle TIFs im flachen Ordner durchgehen
    for filepath in sorted(root_path.glob("*.tif")):
        sample_id = filepath.name
        
        # Eure Logik zur Bestimmung des Labels
        if sample_id.startswith('pos') or sample_id.startswith('ext_pos'):
            label = 1
        elif sample_id.startswith('neg') or sample_id.startswith('ext_neg'):
            label = 0
        else:
            continue  # Überspringe unbekannte Formate
            
        # NoData-Check
        if not check_nodata_ratio(filepath, threshold=0.5):
            all_valid_by_class[label].append(sample_id)
        else:
            print(f"Entfernt (NoData > 50%): {sample_id}")

    # Aufteilen in Train und Val
    train_clean = []
    val_clean = []
    
    for label, samples in all_valid_by_class.items():
        rng = np.random.default_rng(42)  # Fester Seed
        rng.shuffle(samples)
        
        val_size = int(len(samples) * args.val_split)
        val_clean.extend(samples[:val_size])
        train_clean.extend(samples[val_size:])

    splits_dir = Path("splits")
    splits_dir.mkdir(exist_ok=True)
    
    with open(splits_dir / "train_clean.txt", "w") as f:
        for s in sorted(train_clean): f.write(f"{s}\n")
        
    with open(splits_dir / "val_clean.txt", "w") as f:
        for s in sorted(val_clean): f.write(f"{s}\n")

    print(f"\nErgebnis:")
    print(f"  -> splits/train_clean.txt erstellt ({len(train_clean)} Samples)")
    print(f"  -> splits/val_clean.txt erstellt ({len(val_clean)} Samples)")