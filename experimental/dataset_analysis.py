#!/usr/bin/env python3
"""
Dataset Analysis Script

This script analyzes the data_v2 and data_v3 datasets by creating 2x2 grid plots:
- First column: Normalized 255-bin cumulative histogram of images
- Second column: Scatter plot of image mean vs image std
- First row: Train set
- Second row: Validation set

Usage:
    python dataset_analysis.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from tqdm import tqdm
import argparse


def load_image_stats(image_path):
    """
    Load an image and compute its mean and standard deviation.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (mean, std) of the image
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Compute mean and std across all channels
        mean = np.mean(img)
        std = np.std(img)
        
        return mean, std
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None, None


def get_image_files(directory):
    """
    Get only render PNG image files from a directory.
    
    Args:
        directory (str): Directory path
        
    Returns:
        list: List of render image file paths
    """
    # Only get render images (target images)
    image_files = list(Path(directory).glob('*_render_0095.png'))
    return sorted(image_files)


def compute_histogram_data(image_files, num_bins=255, max_images=None):
    """
    Compute histogram data from a list of image files.
    
    Args:
        image_files (list): List of image file paths
        num_bins (int): Number of histogram bins
        max_images (int): Maximum number of images to process (None for all)
        
    Returns:
        tuple: (histogram_data, mean_std_data)
    """
    histogram_data = []
    mean_std_data = []
    
    # Limit number of images if specified
    if max_images is not None:
        image_files = image_files[:max_images]
    
    print(f"Processing {len(image_files)} images...")
    
    for img_path in tqdm(image_files, desc="Computing statistics"):
        # Load image and compute histogram
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Compute histogram for grayscale image
            hist, _ = np.histogram(img.flatten(), bins=num_bins, range=(0, 256))
            
            # Normalize histogram
            hist = hist / np.sum(hist)
            histogram_data.append(hist)
            
            # Compute mean and std
            mean = np.mean(img)
            std = np.std(img)
            mean_std_data.append((mean, std))
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    return histogram_data, mean_std_data


def create_dataset_plots(dataset_name, train_dir, val_dir, output_dir):
    """
    Create 2x2 grid plots for a dataset.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'data_v2')
        train_dir (str): Path to training data directory
        val_dir (str): Path to validation data directory
        output_dir (str): Output directory for plots
    """
    print(f"\nAnalyzing {dataset_name}...")
    
    # Get image files
    train_files = get_image_files(train_dir)
    val_files = get_image_files(val_dir)
    
    if not train_files:
        print(f"No images found in {train_dir}")
        return
    
    if not val_files:
        print(f"No images found in {val_dir}")
        return
    
    print(f"Found {len(train_files)} training images and {len(val_files)} validation images")
    
    # Compute statistics
    train_hist_data, train_mean_std = compute_histogram_data(train_files)
    val_hist_data, val_mean_std = compute_histogram_data(val_files)
    
    if not train_hist_data or not val_hist_data:
        print("No valid histogram data computed")
        return
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Dataset Analysis: {dataset_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Train set normalized histogram
    ax1 = axes[0, 0]
    train_hist_array = np.array(train_hist_data)
    train_mean_histogram = np.mean(train_hist_array, axis=0)
    train_std_histogram = np.std(train_hist_array, axis=0)
    
    x_bins = np.arange(255)
    ax1.plot(x_bins, train_mean_histogram, 'b-', linewidth=2)
    ax1.set_title('Train Set: Normalized Histogram')
    ax1.set_xlabel('Pixel Value')
    ax1.set_ylabel('Probability Density')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Train set mean vs std scatter
    ax2 = axes[0, 1]
    train_means, train_stds = zip(*train_mean_std)
    ax2.scatter(train_means, train_stds, alpha=0.6, s=20, color='blue')
    ax2.set_title('Train Set: Mean vs Standard Deviation')
    ax2.set_xlabel('Image Mean')
    ax2.set_ylabel('Image Standard Deviation')
    ax2.set_xlim(0, 255)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Validation set normalized histogram
    ax3 = axes[1, 0]
    val_hist_array = np.array(val_hist_data)
    val_mean_histogram = np.mean(val_hist_array, axis=0)
    
    ax3.plot(x_bins, val_mean_histogram, 'r-', linewidth=2)
    ax3.set_title('Validation Set: Normalized Histogram')
    ax3.set_xlabel('Pixel Value')
    ax3.set_ylabel('Probability Density')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Validation set mean vs std scatter
    ax4 = axes[1, 1]
    val_means, val_stds = zip(*val_mean_std)
    ax4.scatter(val_means, val_stds, alpha=0.6, s=20, color='red')
    ax4.set_title('Validation Set: Mean vs Standard Deviation')
    ax4.set_xlabel('Image Mean')
    ax4.set_ylabel('Image Standard Deviation')
    ax4.set_xlim(0, 255)
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / f'{dataset_name}_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    
    # Print summary statistics
    print(f"\n{dataset_name} Summary Statistics:")
    print(f"Train set: {len(train_hist_data)} images")
    print(f"  Mean pixel value: {np.mean(train_means):.2f} ± {np.std(train_means):.2f}")
    print(f"  Mean std: {np.mean(train_stds):.2f} ± {np.std(train_stds):.2f}")
    print(f"Validation set: {len(val_hist_data)} images")
    print(f"  Mean pixel value: {np.mean(val_means):.2f} ± {np.std(val_means):.2f}")
    print(f"  Mean std: {np.mean(val_stds):.2f} ± {np.std(val_stds):.2f}")
    
    plt.close()


def main():
    """Main function to analyze both datasets."""
    parser = argparse.ArgumentParser(description='Analyze datasets with 2x2 grid plots')
    parser.add_argument('--output-dir', default='output/dataset_analysis', 
                       help='Output directory for plots (default: experimental)')
    parser.add_argument('--data-v2-dir', default='data_v2',
                       help='Path to data_v2 directory (default: data_v2)')
    parser.add_argument('--data-v3-dir', default='data_v3',
                       help='Path to data_v3 directory (default: data_v3)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Define dataset paths
    datasets = {
        'data_v2': {
            'train': Path(args.data_v2_dir) / 'train',
            'val': Path(args.data_v2_dir) / 'val'
        },
        'data_v3': {
            'train': Path(args.data_v3_dir) / 'train',
            'val': Path(args.data_v3_dir) / 'val'
        }
    }
    
    # Analyze each dataset
    for dataset_name, paths in datasets.items():
        train_dir = paths['train']
        val_dir = paths['val']
        
        if not train_dir.exists():
            print(f"Warning: Training directory {train_dir} does not exist")
            continue
            
        if not val_dir.exists():
            print(f"Warning: Validation directory {val_dir} does not exist")
            continue
        
        create_dataset_plots(dataset_name, train_dir, val_dir, output_dir)
    
    print(f"\nAnalysis complete! Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
