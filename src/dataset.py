"""Custom dataset for 2-band SAR GeoTIFF images using filename labels."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class SARDataset(Dataset):
    """PyTorch Dataset for binary-labelled 2-band SAR GeoTIFF images using filename-based labels."""

    def __init__(
        self,
        root: str,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        split_file: Optional[str] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.split_file = Path(split_file) if split_file else None
        self.samples: List[Tuple[Path, int]] = self._collect_samples()

        if not self.samples:
            raise FileNotFoundError(f"Keine passenden GeoTIFF-Dateien unter '{self.root}' gefunden.")

    def _get_label_from_filename(self, filename: str) -> Optional[int]:
        """Eure Logik zur Label-Bestimmung."""
        if filename.startswith('pos') or filename.startswith('ext_pos'):
            return 1
        elif filename.startswith('neg') or filename.startswith('ext_neg'):
            return 0
        return None

    def _collect_samples(self) -> List[Tuple[Path, int]]:
        samples: List[Tuple[Path, int]] = []
        
        if not self.split_file.is_file():
            raise FileNotFoundError(f"Bereinigte Split-Datei nicht gefunden: {self.split_file}")
            
        with open(self.split_file, "r") as f:
            filenames = [line.strip() for line in f if line.strip()]
            
        for fname in filenames:
            # HIER DIE KORREKTUR: Wenn der Eintrag aus dem Split nicht auf '_s1.tif' endet,
            # bauen wir den korrekten Dateinamen für die Festplatte zusammen.
            if not fname.endswith('_s1.tif'):
                clean_fname = fname.replace('.tif', '')
                real_file = f"{clean_fname}_s1.tif"
            else:
                real_file = fname
                
            filepath = self.root / real_file
            label = self._get_label_from_filename(fname)
            if label is not None:
                samples.append((filepath, label))
                
        return samples

        # Fall 2: Fallback (alles im Ordner einlesen)
        for filepath in sorted(self.root.glob("*.tif")):
            label = self._get_label_from_filename(filepath.name)
            if label is not None:
                samples.append((filepath, label))
        return samples

    @staticmethod
    def _load_geotiff(filepath: Path) -> torch.Tensor:
        with rasterio.open(filepath) as src:
            if src.count != 2:
                raise ValueError(f"Expected 2 bands (VV, VH) in '{filepath}', got {src.count}.")
            data = src.read().astype(np.float32)
        return torch.from_numpy(data)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, label = self.samples[idx]
        image = self._load_geotiff(filepath)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def class_counts(self) -> Dict[int, int]:
        counts = {0: 0, 1: 0}
        for _, label in self.samples:
            counts[label] += 1
        return counts