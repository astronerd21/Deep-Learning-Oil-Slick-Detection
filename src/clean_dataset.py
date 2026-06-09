"""Script to clean existing SAR image splits by filtering out high NoData content."""

import argparse
from pathlib import Path
import rasterio
import numpy as np


def check_nodata_ratio(filepath: Path, threshold: float) -> bool:
    """Calculate the percentage of pixels with exactly 0.0 or NaN in the VV channel."""
    if not filepath.is_file():
        return None

    with rasterio.open(filepath) as src:
        vv_band = src.read(1)

    nodata_pixels = np.sum((vv_band == 0.0) | (np.isnan(vv_band)))
    total_pixels = vv_band.size
    return (nodata_pixels / total_pixels) > threshold


def clean_single_split_file(
    original_split_path: Path, output_split_path: Path, data_dir: Path, threshold: float
):
    """Iterate over current split IDs, filter out artifacts, and save new splits."""
    if not original_split_path.is_file():
        print(f"Warning: Original split file not found at {original_split_path}")
        return

    valid_image_ids = []
    removed_count = 0
    missing_count = 0

    with open(original_split_path, "r") as f:
        image_ids = [line.strip() for line in f if line.strip()]

    for img_id in image_ids:
        if not img_id.endswith("_s1.tif"):
            clean_id = img_id.replace(".tif", "")
            img_file = f"{clean_id}_s1.tif"
        else:
            img_file = img_id

        img_path = data_dir / img_file

        nodata_check = check_nodata_ratio(img_path, threshold=threshold)

        if nodata_check is None:
            missing_count += 1
        elif not nodata_check:
            valid_image_ids.append(img_id)
        else:
            removed_count += 1

    output_split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_split_path, "w") as f:
        for img_id in sorted(valid_image_ids):
            f.write(f"{img_id}\n")

    split_name = original_split_path.name.upper()
    print(
        f" -> {split_name}: {len(valid_image_ids):>4} valid | {removed_count:>3} artifacts removed | {missing_count:>3} not found"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean current splits based on VV NoData threshold."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--splits-in-dir", required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Threshold for NoData filtering (e.g. 0.2 for 20%)",
    )
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    source_splits_dir = Path(args.splits_in_dir)
    output_splits_dir = Path("/content/cleaned_splits")

    print(f"{'Split Name':<13} | {'Status':<68}")
    print(f"Using Threshold: {args.threshold * 100}%")
    print("-" * 85)
    clean_single_split_file(
        source_splits_dir / "train.txt",
        output_splits_dir / "train_clean.txt",
        data_path,
        args.threshold,
    )
    clean_single_split_file(
        source_splits_dir / "val.txt",
        output_splits_dir / "val_clean.txt",
        data_path,
        args.threshold,
    )
    print("-" * 85)
