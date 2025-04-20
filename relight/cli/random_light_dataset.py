#!/usr/bin/env python
"""
Random Light Dataset CLI

This script provides a command-line interface for generating random light datasets.
"""

import argparse
import bpy

from relight.core.random_light_dataset import generate_random_light_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Random Light Dataset CLI")
    
    parser.add_argument("--start-index", type=int, default=0, help="Starting index for the dataset")
    parser.add_argument("--n-images", type=int, default=100, help="Number of images to generate")
    parser.add_argument("--no-gpu", action="store_true", help="Don't use GPU rendering")
    parser.add_argument("--no-progress", action="store_true", help="Don't show progress")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Generate random light dataset
    generate_random_light_dataset(
        start_index=args.start_index,
        n_images=args.n_images,
        use_gpu=not args.no_gpu,
        show_progress=not args.no_progress
    )
    
    print("Random light dataset generation completed successfully.")


if __name__ == "__main__":
    main() 