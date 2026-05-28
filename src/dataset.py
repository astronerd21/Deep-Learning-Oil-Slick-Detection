"""Custom dataset for 2-band SAR GeoTIFF images (VV/VH polarisations)."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class SARDataset(Dataset):
    """PyTorch Dataset for binary-labelled 2-band SAR GeoTIFF images.

    Unterstützt das Laden des gesamten Ordners oder das Filtern über eine vordefinierte Split-Datei.
    """

    #: Mapping from sub-directory name to integer class label.
    CLASSES = {"no_oil_slick": 0, "oil_slick": 1}

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
            raise FileNotFoundError(
                f"No GeoTIFF files found under '{self.root}' with the given configuration."
            )

    def _collect_samples(self) -> List[Tuple[Path, int]]:
        samples: List[Tuple[Path, int]] = []
        
        # Fall 1: Nutzen der bereinigten Split-Textdatei
        if self.split_file:
            if not self.split_file.is_file():
                raise FileNotFoundError(f"Split-Datei nicht gefunden: {self.split_file}")
                
            with open(self.split_file, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
                
            for rel_path in lines:
                filepath = self.root / rel_path
                # Bestimme das Label anhand des Verzeichnisnamens im relativen Pfad
                class_name = Path(rel_path).parts[0]
                if class_name in self.CLASSES:
                    samples.append((filepath, self.CLASSES[class_name]))
            return samples

        # Fall 2: Fallback (originaler Modus) scannt das komplette Verzeichnis
        for class_name, label in self.CLASSES.items():
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            for ext in ("*.tif", "*.tiff"):
                for filepath in sorted(class_dir.glob(ext)):
                    samples.append((filepath, label))
        return samples

    @staticmethod
    def _load_geotiff(filepath: Path) -> torch.Tensor:
        """Read a 2-band GeoTIFF and return a ``(2, H, W)`` float32 tensor."""
        with rasterio.open(filepath) as src:
            if src.count != 2:
                raise ValueError(
                    f"Expected 2 bands (VV, VH) in '{filepath}', "
                    f"got {src.count}."
                )
            data = src.read().astype(np.float32)  # (2, H, W)
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
        """Return the number of samples per class label."""
        counts: Dict[int, int] = {label: 0 for label in self.CLASSES.values()}
        for _, label in self.samples:
            counts[label] += 1
        return counts