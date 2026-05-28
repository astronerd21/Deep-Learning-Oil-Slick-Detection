"""Script to clean SAR image chips by filtering out high NoData content."""

import argparse
from pathlib import Path
import rasterio
import numpy as np


def check_nodata_ratio(filepath: Path, threshold: float = 0.5) -> bool:
    """Returns True if the VV band contains more 0.0 pixels than the threshold."""
    with rasterio.open(filepath) as src:
        # Band 1 ist der VV-Kanal laut Dataset-Spezifikation
        vv_band = src.read(1)
        
    nodata_pixels = np.sum(vv_band == 0.0)
    total_pixels = vv_band.size
    nodata_ratio = nodata_pixels / total_pixels
    
    return nodata_ratio > threshold


def clean_split(data_dir: Path, out_file: Path, threshold: float = 0.5):
    """Iterates through oil_slick and no_oil_slick to find and filter valid files."""
    print(f"Starte Bereinigung für Datenverzeichnis: {data_dir}")
    valid_samples = []
    skipped_count = 0
    
    # Klassen-Unterordner wie im originalen SARDataset definiert
    classes = ["no_oil_slick", "oil_slick"]
    
    for class_name in classes:
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            continue
            
        for ext in ("*.tif", "*.tiff"):
            for filepath in sorted(class_dir.glob(ext)):
                is_corrupted = check_nodata_ratio(filepath, threshold)
                
                # Speichere relativen Pfad für die Split-Datei
                rel_path = filepath.relative_to(data_dir)
                
                if not is_corrupted:
                    valid_samples.append(str(rel_path))
                else:
                    skipped_count += 1

    # Schreibe die gültigen IDs/Pfade in die Textdatei
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        for sample in sorted(valid_samples):
            f.write(f"{sample}\n")
            
    print(f"Bereinigung abgeschlossen für {out_file.name}:")
    print(f"  - Gültige Bilder: {len(valid_samples)}")
    print(f"  - Ausgeschlossene Bilder (> {threshold*100}% NoData): {skipped_count}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter out SAR chips with high NoData edge artifacts.")
    parser.add_argument("--data-dir", required=True, help="Root-Pfad zum Datensatz-Ordner.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Anteil der Daten für Validierung.")
    args = parser.parse_args()

    root_path = Path(args.data_dir)
    
    # Da das aktuelle Projekt noch keine fixen train/val-Ordner besitzt,
    # sammeln wir die validen Daten und splitten sie hier deterministisch auf.
    classes = ["no_oil_slick", "oil_slick"]
    all_valid_by_class = {0: [], 1: []}
    
    for label, class_name in enumerate(classes):
        class_dir = root_path / class_name
        if not class_dir.is_dir():
            continue
        for ext in ("*.tif", "*.tiff"):
            for filepath in sorted(class_dir.glob(ext)):
                if not check_nodata_ratio(filepath, threshold=0.5):
                    all_valid_by_class[label].append(str(filepath.relative_to(root_path)))
                else:
                    print(f"Entfernt (NoData > 50%): {filepath.name}")

    # Aufteilen in Train und Val unter Beibehaltung der Schichtung (Stratification)
    train_clean = []
    val_clean = []
    
    for label, samples in all_valid_by_class.items():
        # Festes Seeding für Reproduzierbarkeit des Splits
        rng = np.random.default_rng(42)
        rng.shuffle(samples)
        
        val_size = int(len(samples) * args.val_split)
        val_clean.extend(samples[:val_size])
        train_clean.extend(samples[val_size:])

    # Speicher die finalen .txt Dateien im Projekt-Root oder unter splits/
    splits_dir = Path("splits")
    splits_dir.mkdir(exist_ok=True)
    
    with open(splits_dir / "train_clean.txt", "w") as f:
        for s in sorted(train_clean): f.write(f"{s}\n")
        
    with open(splits_dir / "val_clean.txt", "w") as f:
        for s in sorted(val_clean): f.write(f"{s}\n")

    print(f"\nErgebnis:")
    print(f"  -> splits/train_clean.txt erstellt ({len(train_clean)} Samples)")
    print(f"  -> splits/val_clean.txt erstellt ({len(val_clean)} Samples)")