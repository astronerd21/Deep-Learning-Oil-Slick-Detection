"""Custom dataset for 2-band SAR GeoTIFF images using existing clean splits."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class SARDataset(Dataset):
    """PyTorch Dataset that loads valid SAR image chips from a clean split file."""

    def __init__(
        self,
        root: str,
        split_file: str,
        metadata: Optional[str] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.root = Path(root)
        self.split_file = Path(split_file)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = self._collect_samples()

        if not self.samples:
            raise FileNotFoundError(f"No valid GeoTIFF files loaded from split file '{self.split_file}'.")

    def _get_label_from_filename(self, filename: str) -> Optional[int]:
        """Derive binary label from the filename prefix."""
        if filename.startswith('pos') or filename.startswith('ext_pos'):
            return 1
        elif filename.startswith('neg') or filename.startswith('ext_neg'):
            return 0
        return None

    def _collect_samples(self) -> List[Tuple[Path, int]]:
        samples: List[Tuple[Path, int]] = []
        
        if not self.split_file.is_file():
            raise FileNotFoundError(f"Cleaned split file not found: {self.split_file}")
            
        with open(self.split_file, "r") as f:
            filenames = [line.strip() for line in f if line.strip()]
            
        for fname in filenames:
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

    @staticmethod
    def _load_geotiff(filepath: Path) -> torch.Tensor:
        """Loads a 2-band GeoTIFF and applies channel-wise Z-score normalization."""
        with rasterio.open(filepath) as src:
            if src.count != 2:
                raise ValueError(f"Expected 2 bands (VV, VH) in '{filepath}', got {src.count}.")
            data = src.read().astype(np.float32)  # Shape: (2, H, W)
        
        # Apply Z-score normalization per channel (band) to stabilize radar values
        for c in range(data.shape[0]):
            channel_mean = np.mean(data[c])
            channel_std = np.std(data[c])
            if channel_std > 0:
                data[c] = (data[c] - channel_mean) / channel_std
                
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