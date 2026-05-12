import os
import torch
import rasterio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class SARDataset(Dataset):
    def __init__(self, data_dir, split_file, metadata_file, transform=None):
        """
        Initializes the Sentinel-1 SAR Dataset.
        
        Args:
            data_dir (str): Path to the extracted images (e.g., '/content/data/images_s1')
            split_file (str): Path to the split txt file (e.g., 'splits/random/train.txt')
            metadata_file (str): Path to metadata.csv
            transform (callable, optional): Optional torchvision transforms
        """

        self.data_dir = data_dir
        self.transform = transform

        # 1. Read IDs from split file
        with open(split_file, 'r') as f:
            self.sample_ids = [line.strip() for line in f.readlines() if line.strip()]

        # 2. Load metadata.csv to obtain labels
        self.metadata = pd.read_csv(metadata_file)

        # Filter metadata to only include items present in the current split
        self.metadata = self.metadata[self.metadata['sample_id'].isin(self.sample_ids)]
        self.metadata.set_index('sample_id', inplace=True)

         # Safety check: Verify that the physical files actually exist
        valid_ids = []
        for sid in self.sample_ids:
            img_path = os.path.join(self.data_dir, f"{sid}_s1.tif")
            if os.path.exists(img_path):
                valid_ids.append(sid)
        self.sample_ids = valid_ids

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # 3. Load image from the central images_s1/ folder
        img_path = os.path.join(self.data_dir, f"{sample_id}_s1.tif")
        
        # Extract label ('pos' or 'ext_pos' = 1, 'neg' or 'ext_neg' = 0)
        label_str = str(self.metadata.loc[sample_id, 'label'])
        label = 1.0 if 'pos' in label_str else 0.0
        
        # Open GeoTIFF using rasterio (reads VV and VH backscatter)
        with rasterio.open(img_path) as src:
            vv = src.read(1) # Band 1
            vh = src.read(2) # Band 2
            
        # Convert to PyTorch Tensor (Shape: Channels, Height, Width)
        image = torch.tensor(np.stack([vv, vh]), dtype=torch.float32)
        
        # Z-Score Normalization per channel
        for c in range(image.shape[0]):
            mean = image[c].mean()
            std = image[c].std()
            if std > 0:
                image[c] = (image[c] - mean) / std
        
        # Apply any augmentations if provided
        if self.transform:
            image = self.transform(image)
            
        # Binary Cross-Entropy loss in PyTorch usually expects float32 labels
        return image, torch.tensor(label, dtype=torch.float32)