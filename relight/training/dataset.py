"""
Dataset module for Relight training.

This module provides dataset classes for loading and preprocessing image pairs
for training ControlNet and other models.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class RelightDataset(Dataset):
    """
    Dataset for loading pairs of control and target images for training.
    Supports loading from a CSV file that specifies the subset to use.
    """
    
    def __init__(self, 
                 data_dir: str,
                 subset_size: Optional[int] = None,
                 control_transform: Optional[transforms.Compose] = None,
                 target_transform: Optional[transforms.Compose] = None,
                 image_size: int = 512,
                 normalize_images: bool = True):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing both control and target images and CSVs
            subset_size: Number of images in the subset to load (e.g., 1000 for light_positions_1000.csv)
            control_transform: Transform to apply to control images
            target_transform: Transform to apply to target images
            image_size: Size to resize images to
            normalize_images: Whether to normalize images to [-1, 1]
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.subset_size = subset_size
        csv_path = self.data_dir / f"light_positions_{subset_size}.csv" if subset_size else self.data_dir / "light_positions.csv"
        if not csv_path.exists():
            raise ValueError(f"Subset CSV {csv_path} not found.")
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            self.entries = list(reader)
        if not self.entries:
            raise ValueError(f"No entries found in {csv_path}")
        
        # Default transforms if none provided
        if control_transform is None:
            control_transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])
            if normalize_images:
                control_transform.transforms.append(transforms.Normalize([0.5], [0.5]))
                
        if target_transform is None:
            target_transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])
            if normalize_images:
                target_transform.transforms.append(transforms.Normalize([0.5], [0.5]))
        
        self.control_transform = control_transform
        self.target_transform = target_transform
        logger.info(f"Loaded {len(self.entries)} image pairs from {csv_path}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.entries)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Dictionary containing control and target images
        """
        entry = self.entries[idx]
        # The 'index' field in the CSV is the image base name
        base_name = int(entry['index'])
        # Compose file names
        target_path = self.data_dir / f"{base_name:05d}_render_0095.png"
        control_path = self.data_dir / f"{base_name:05d}_diffdir_0095.png"
        # Load images
        control_image = Image.open(control_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")
        # Apply transforms
        control_image = self.control_transform(control_image)
        target_image = self.target_transform(target_image)
        return {
            'conditioning_pixel_values': control_image,
            'pixel_values': target_image,
            'control_file': control_path.name,
            'target_file': target_path.name,
            'meta': entry
        }


def main():
    """Debug function to visualize dataset samples."""
    import matplotlib.pyplot as plt
    import torch
    from torchvision.utils import make_grid

    # Create dataset
    dataset = RelightDataset(
        data_dir="/home/dcor/orkozlovsky/repos/relight/data_v3/train",
        subset_size=5438,
        image_size=768,
        normalize_images=True
    )
    
    # Get 25 random samples
    num_samples = 25
    total_len = len(dataset)
    torch.manual_seed(42)  # For reproducibility
    indices = torch.randperm(total_len)[:num_samples].tolist()
    samples = []
    for i in indices:
        sample = dataset[i]
        # Denormalize images
        control = (sample['conditioning_pixel_values'] + 1) / 2
        target = (sample['pixel_values'] + 1) / 2
        # Stack control and target horizontally
        pair = torch.cat([control, target], dim = 2)
        samples.append(pair)
    
    # Make 5x5 grid
    grid = make_grid(samples, nrow = 5, padding = 20, pad_value = 1)  # Add padding between grid items
    
    # Convert to numpy and transpose for plotting
    grid = grid.numpy().transpose(1, 2, 0)
    
    # Plot with figure size matching grid dimensions
    h, w = grid.shape[:2]
    aspect_ratio = w / h
    plt.figure(figsize=(10 * aspect_ratio, 10))
    plt.imshow(grid)
    plt.axis('off')
    plt.savefig('/home/dcor/orkozlovsky/repos/relight/data_v3/trainset_5438_samples.png')
    plt.close()

if __name__ == "__main__":
    main()
