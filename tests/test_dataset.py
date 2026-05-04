"""Unit tests for src.dataset.SARDataset."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.dataset import SARDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_geotiff_tree(root: Path) -> None:
    """Create the expected directory structure with dummy .tif file names."""
    (root / "oil_slick").mkdir(parents=True)
    (root / "no_oil_slick").mkdir(parents=True)
    for name in ["a.tif", "b.tif"]:
        (root / "oil_slick" / name).touch()
    for name in ["c.tif"]:
        (root / "no_oil_slick" / name).touch()


def _mock_rasterio_open(bands: int = 2, height: int = 64, width: int = 64):
    """Return a context manager mock that pretends to be an open rasterio file."""
    mock_src = MagicMock()
    mock_src.count = bands
    mock_src.read.return_value = np.random.rand(bands, height, width).astype(np.float32)
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_src)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSARDatasetInit:
    def test_finds_tif_files(self, tmp_path):
        _make_geotiff_tree(tmp_path)
        with patch("rasterio.open", return_value=_mock_rasterio_open()):
            ds = SARDataset(root=str(tmp_path))
        assert len(ds) == 3  # 2 oil_slick + 1 no_oil_slick

    def test_raises_on_empty_root(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SARDataset(root=str(tmp_path))

    def test_missing_class_dir_is_tolerated(self, tmp_path):
        """If only one class dir exists the dataset still loads."""
        (tmp_path / "oil_slick").mkdir()
        (tmp_path / "oil_slick" / "scene.tif").touch()
        with patch("rasterio.open", return_value=_mock_rasterio_open()):
            ds = SARDataset(root=str(tmp_path))
        assert len(ds) == 1

    def test_correct_labels(self, tmp_path):
        _make_geotiff_tree(tmp_path)
        with patch("rasterio.open", return_value=_mock_rasterio_open()):
            ds = SARDataset(root=str(tmp_path))
        labels = [label for _, label in ds.samples]
        assert set(labels) == {0, 1}

    def test_class_counts(self, tmp_path):
        _make_geotiff_tree(tmp_path)
        with patch("rasterio.open", return_value=_mock_rasterio_open()):
            ds = SARDataset(root=str(tmp_path))
        counts = ds.class_counts
        assert counts[1] == 2   # oil_slick
        assert counts[0] == 1   # no_oil_slick


class TestSARDatasetGetItem:
    def _make_ds(self, tmp_path, transform=None):
        _make_geotiff_tree(tmp_path)
        with patch("rasterio.open", return_value=_mock_rasterio_open()):
            return SARDataset(root=str(tmp_path), transform=transform)

    def test_returns_tensor_and_int(self, tmp_path):
        ds = self._make_ds(tmp_path)
        with patch("rasterio.open", return_value=_mock_rasterio_open()):
            image, label = ds[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)

    def test_image_shape(self, tmp_path):
        ds = self._make_ds(tmp_path)
        with patch("rasterio.open", return_value=_mock_rasterio_open(2, 64, 64)):
            image, _ = ds[0]
        assert image.shape == torch.Size([2, 64, 64])

    def test_image_dtype(self, tmp_path):
        ds = self._make_ds(tmp_path)
        with patch("rasterio.open", return_value=_mock_rasterio_open()):
            image, _ = ds[0]
        assert image.dtype == torch.float32

    def test_transform_is_applied(self, tmp_path):
        called = {"flag": False}

        def my_transform(t):
            called["flag"] = True
            return t * 2.0

        ds = self._make_ds(tmp_path, transform=my_transform)
        with patch("rasterio.open", return_value=_mock_rasterio_open()):
            ds[0]
        assert called["flag"]

    def test_wrong_band_count_raises(self, tmp_path):
        _make_geotiff_tree(tmp_path)
        with patch("rasterio.open", return_value=_mock_rasterio_open()):
            ds = SARDataset(root=str(tmp_path))
        with patch("rasterio.open", return_value=_mock_rasterio_open(bands=3)):
            with pytest.raises(ValueError, match="Expected 2 bands"):
                ds[0]

    def test_tiff_extension_also_found(self, tmp_path):
        (tmp_path / "oil_slick").mkdir(parents=True)
        (tmp_path / "oil_slick" / "scene.tiff").touch()
        (tmp_path / "no_oil_slick").mkdir()
        (tmp_path / "no_oil_slick" / "other.tif").touch()
        with patch("rasterio.open", return_value=_mock_rasterio_open()):
            ds = SARDataset(root=str(tmp_path))
        assert len(ds) == 2
