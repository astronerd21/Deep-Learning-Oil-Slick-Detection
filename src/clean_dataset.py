"""Script to clean existing SAR image splits by filtering out high NoData content."""

import argparse
from pathlib import Path
import rasterio
import numpy as np


def check_nodata_ratio(filepath: Path, threshold: float = 0.5) -> bool:
    """Calculate the percentage of pixels with exactly 0.0, NaN or artificial NoData (-163.0) in the VV channel."""
    if not filepath.is_file():
        return None

    with rasterio.open(filepath) as src:
        vv_band = src.read(1)

    nodata_pixels = np.sum((vv_band == 0.0) | (np.isnan(vv_band)) | (vv_band <= -160.0))
    total_pixels = vv_band.size

    return (nodata_pixels / total_pixels) > threshold


def clean_single_split_file(
    original_split_path: Path, output_split_path: Path, data_dir: Path
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
        base_name = img_id.split(",")[0]

        if not base_name.endswith("_s1.tif"):
            clean_id = base_name.replace(".tif", "")
            img_file = f"{clean_id}_s1.tif"
        else:
            img_file = base_name

        img_path = data_dir / img_file

        nodata_check = check_nodata_ratio(img_path, threshold=0.5)

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
        f" -> {split_name:<10}: {len(valid_image_ids):>4} valid | {removed_count:>3} artifacts removed | {missing_count:>3} not found"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean current splits based on VV NoData threshold."
    )
    parser.add_argument(
        "--data-dir", required=True, help="Directory containing the TIFF images."
    )
    parser.add_argument(
        "--splits-in-dir",
        required=True,
        help="Directory containing the source split .txt files.",
    )
    parser.add_argument(
        "--out-prefix",
        default="",
        help="Optional prefix for the output files (e.g., 'geo_').",
    )
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    source_splits_dir = Path(args.splits_in_dir)
    output_splits_dir = Path("/content/cleaned_splits")

    if not source_splits_dir.is_dir():
        print(f"Error: Directory {source_splits_dir} does not exist.")
        exit(1)

    print(f"{'Split Name':<16} | {'Status':<68}")
    print("-" * 85)

    for split_file in sorted(source_splits_dir.glob("*.txt")):
        output_filename = f"{args.out_prefix}{split_file.stem}_clean.txt"
        output_file_path = output_splits_dir / output_filename

        clean_single_split_file(
            original_split_path=split_file,
            output_split_path=output_file_path,
            data_dir=data_path,
        )

    print("-" * 85)
