"""
Dataset module for Relight training.

This module provides dataset classes for loading and preprocessing image pairs
for training ControlNet and other models.
"""

from typing import Optional, Dict
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class RelightDataset(Dataset):
    """
    Dataset for loading pairs of control and target images for training.
    
    This dataset loads control images and target images from the same directory
    with specific naming patterns:
    - Target images: XXXXX_render_00000.jpg
    - Control images: XXXXX_diffdir_00000.jpg
    where XXXXX is a 5-digit index starting from 0.
    """
    
    def __init__(self, 
                 data_dir: str,
                 control_transform: Optional[transforms.Compose] = None,
                 target_transform: Optional[transforms.Compose] = None,
                 image_size: int = 512,
                 normalize_images: bool = True):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing both control and target images
            control_transform: Transform to apply to control images
            target_transform: Transform to apply to target images
            image_size: Size to resize images to
            normalize_images: Whether to normalize images to [-1, 1]
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
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
        
        # Get list of target image files
        self.image_files = sorted([f for f in self.data_dir.glob("*_render_0000.png")])
        if not self.image_files:
            raise ValueError(f"No target image files found in {data_dir}")
                
        logger.info(f"Loaded {len(self.image_files)} image pairs")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Dictionary containing control and target images
        """
        # Get base filename (XXXXX)
        target_path = self.image_files[idx]
        base_name = target_path.stem.split('_')[0]
        
        # Construct control image path
        control_path = self.data_dir / f"{base_name}_diffdir_0000.png"
        
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
            'target_file': target_path.name
        }


def main():
    """Debug function to visualize dataset samples."""
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    import torch
    
    # Create dataset
    dataset = RelightDataset(
        data_dir=Path("C:/repos/relight/output"),
        image_size=256,
        normalize_images=True
    )
    
    # Get 25 samples
    samples = []
    for i in range(25):
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
    plt.savefig('C:/repos/relight/output/dataset_samples.png')
    plt.close()

if __name__ == "__main__":
    main()