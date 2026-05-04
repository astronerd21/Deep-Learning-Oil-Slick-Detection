"""Custom dataset for 2-band SAR GeoTIFF images (VV/VH polarisations)."""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class SARDataset(Dataset):
    """PyTorch Dataset for binary-labelled 2-band SAR GeoTIFF images.

    The dataset expects images to be organised under *root* in two
    sub-directories that reflect the class label:

    .. code-block:: text

        root/
          oil_slick/        # label = 1
            scene_001.tif
            scene_002.tif
            ...
          no_oil_slick/     # label = 0
            scene_101.tif
            ...

    Each GeoTIFF must contain exactly two bands ordered as
    **VV (band 1)** and **VH (band 2)**.

    Args:
        root: Path to the root directory that contains the class folders.
        transform: Optional callable applied to the ``(2, H, W)``
            ``torch.FloatTensor`` before it is returned.  Use this hook for
            normalisation, cropping, or any other augmentation.
    """

    #: Mapping from sub-directory name to integer class label.
    CLASSES = {"no_oil_slick": 0, "oil_slick": 1}

    def __init__(
        self,
        root: str,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = self._collect_samples()

        if not self.samples:
            raise FileNotFoundError(
                f"No GeoTIFF files found under '{self.root}'. "
                "Expected sub-directories: "
                + ", ".join(f"'{k}'" for k in self.CLASSES)
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_samples(self) -> List[Tuple[Path, int]]:
        samples: List[Tuple[Path, int]] = []
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

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, label = self.samples[idx]
        image = self._load_geotiff(filepath)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def class_counts(self) -> dict:
        """Return the number of samples per class label."""
        counts: dict = {label: 0 for label in self.CLASSES.values()}
        for _, label in self.samples:
            counts[label] += 1
        return counts
