"""Script to clean existing SAR image splits by filtering out high NoData content."""

import argparse
from pathlib import Path
import rasterio
import numpy as np

def check_nodata_ratio(filepath: Path, threshold: float = 0.5) -> bool:
    """Opens the VV channel (band 1) and calculates the percentage of 0.0 pixels."""
    if not filepath.is_file():
        return True # Falls eine Datei fehlt, wird sie ausgeschlossen
        
    with rasterio.open(filepath) as src:
        # 1. Öffne den VV-Kanal (Band 1)
        vv_band = src.read(1)
        
    # 2. Berechne den Prozentsatz der Pixel mit dem Wert exakt 0.0
    nodata_pixels = np.sum(vv_band == 0.0)
    total_pixels = vv_band.size
    nodata_ratio = nodata_pixels / total_pixels
    
    # 3. Schwellenwert prüfen (> 50%)
    return nodata_ratio > threshold

def clean_single_split_file(original_split_path: Path, output_split_path: Path, data_dir: Path):
    """Iterates over images listed in the current split file and filters them."""
    if not original_split_path.is_file():
        print(f"Warnung: Originale Split-Datei nicht gefunden unter {original_split_path}")
        return

    valid_image_ids = []
    removed_count = 0

    # Iteriere über alle Bilder im aktuellen Split
    with open(original_split_path, "r") as f:
        image_ids = [line.strip() for line in f if line.strip()]

    for img_id in image_ids:
        # Pfad zur Bilddatei zusammenbauen
        img_path = data_dir / img_id
        
        # Wenn NoData > 50%, wird es ausgeschlossen, ansonsten behalten
        if not check_nodata_ratio(img_path, threshold=0.5):
            valid_image_ids.append(img_id)
        else:
            removed_count += 1
           # print(f"  -> Ausgeschlossen (NoData > 50%): {img_id}")

    # Deliverables: Generiere die neue, bereinigte Split-Datei
    output_split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_split_path, "w") as f:
        for img_id in sorted(valid_image_ids):
            f.write(f"{img_id}\n")

    print(f"Datei '{output_split_path.name}' generiert:")
    print(f"  - Gültige Bilder übrig: {len(valid_image_ids)}")
    print(f"  - Entfernte Artefakte: {removed_count}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean current splits based on VV NoData threshold.")
    parser.add_argument("--data-dir", required=True, help="Pfad zum Ordner mit den entpackten .tif Bildern.")
    parser.add_argument("--splits-in-dir", required=True, help="Pfad zum Ordner mit den originalen train.txt und val.txt.")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    source_splits_dir = Path(args.splits_in_dir)
    
    # Zielordner für die neuen Deliverables (train_clean.txt und val_clean.txt)
    output_splits_dir = Path("splits")

    print("=== Starte Bereinigung des TRAINING-Splits ===")
    clean_single_split_file(
        original_split_path=source_splits_dir / "train.txt", 
        output_split_path=output_splits_dir / "train_clean.txt", 
        data_dir=data_path
    )

    print("=== Starte Bereinigung des VALIDATION-Splits ===")
    clean_single_split_file(
        original_split_path=source_splits_dir / "val.txt", 
        output_split_path=output_splits_dir / "val_clean.txt", 
        data_dir=data_path
    )