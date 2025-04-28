"""
Dataset module for Relight training.

This module provides dataset classes for loading and preprocessing image pairs
for training ControlNet and other models.
"""

import os
import json
import random
from typing import Optional, Dict, Any, List, Tuple, Union
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ImagePairDataset(Dataset):
    """
    Dataset for loading pairs of images for training.
    
    This dataset loads input images, control images, and target images
    from specified directories and applies transformations to them.
    """
    
    def __init__(self, 
                 input_dir: str,
                 control_dir: str,
                 target_dir: str,
                 image_size: int = 512,
                 transform: Optional[transforms.Compose] = None,
                 caption_file: Optional[str] = None):
        """
        Initialize the dataset.
        
        Args:
            input_dir: Directory containing input images
            control_dir: Directory containing control images
            target_dir: Directory containing target images
            image_size: Size to resize images to
            transform: Optional custom transform to apply to images
            caption_file: Optional path to a file containing captions for images
        """
        self.input_dir = Path(input_dir)
        self.control_dir = Path(control_dir)
        self.target_dir = Path(target_dir)
        self.image_size = image_size
        
        # Default transform if none provided
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Get list of image files
        self.image_files = sorted([f for f in self.input_dir.glob("*.png")])
        if not self.image_files:
            self.image_files = sorted([f for f in self.input_dir.glob("*.jpg")])
            
        if not self.image_files:
            raise ValueError(f"No image files found in {input_dir}")
            
        # Load captions if provided
        self.captions = {}
        if caption_file:
            with open(caption_file, "r") as f:
                self.captions = json.load(f)
                
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
            Dictionary containing input, control, and target images
        """
        # Get image paths
        input_path = self.image_files[idx]
        control_path = self.control_dir / input_path.name
        target_path = self.target_dir / input_path.name
        
        # Load images
        input_image = Image.open(input_path).convert("RGB")
        control_image = Image.open(control_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")
        
        # Apply transforms
        if self.transform:
            input_image = self.transform(input_image)
            control_image = self.transform(control_image)
            target_image = self.transform(target_image)
        
        # Get caption if available
        caption = self.captions.get(input_path.stem, "")
            
        return {
            'input': input_image,
            'control': control_image,
            'target': target_image,
            'caption': caption,
            'input_file': input_path.name,
            'control_file': control_path.name,
            'target_file': target_path.name
        } 

def get_train_dataset(
    args,
    transform: Optional[transforms.Compose] = None,
) -> Dataset:
    """
    Get a training dataset based on the provided arguments.
    
    Args:
        args: Arguments containing dataset configuration
        transform: Optional transform to apply to images
        
    Returns:
        A dataset for training
    """
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
    if args.input_dir is not None:
        # Use custom dataset with image pairs
        dataset = ImagePairDataset(
            input_dir=args.input_dir,
            control_dir=args.control_dir,
            target_dir=args.target_dir,
            caption_file=args.caption_file,
            transform=transform,
        )
    else:
        # Use HuggingFace dataset
        from datasets import load_dataset
        
        if args.dataset_name is not None:
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
            )
        else:
            dataset = load_dataset(
                "imagefolder",
                data_dir=args.train_data_dir,
                cache_dir=args.cache_dir,
            )
            
        # Preprocess dataset
        def preprocess_function(examples):
            images = [image.convert("RGB") for image in examples[args.image_column]]
            examples["pixel_values"] = [transform(image) for image in images]
            
            if args.conditioning_image_column in examples:
                images = [image.convert("RGB") for image in examples[args.conditioning_image_column]]
                examples["conditioning_pixel_values"] = [transform(image) for image in images]
                
            return examples
            
        with torch.no_grad():
            dataset = dataset.map(
                preprocess_function,
                batched=True,
                batch_size=args.dataset_preprocess_batch_size,
                remove_columns=[col for col in dataset.column_names if col != args.image_column],
            )
            
    return dataset 