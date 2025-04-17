#!/usr/bin/env python
"""
Example: Generating Random Light Dataset

This script demonstrates how to generate a dataset of random light positions in a scene.
"""

import bpy
import argparse
from pathlib import Path

# Add the project root to the Python path
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from relight.core.random_light_dataset import generate_random_light_dataset


def parse_args():
    """Parse command line arguments."""
    # Get the script arguments, ignoring Blender's arguments
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Example: Generating Random Light Dataset")
    parser.add_argument("--start-index", type=int, default=0, help="Starting index for the dataset")
    parser.add_argument("--n-images", type=int, default=10, help="Number of images to generate")
    parser.add_argument("--random-sphere", action="store_true", help="Use random sphere positioning")
    parser.add_argument("--no-gpu", action="store_true", help="Don't use GPU rendering")
    parser.add_argument("--no-progress", action="store_true", help="Don't show progress bar")
    parser.add_argument("--output-dir", type=str, help="Output directory for the generated images")
    return parser.parse_args(argv)


def main():
    """Main function."""
    args = parse_args()
    
    # Set the output directory if provided
    if args.output_dir:
        output_dir = Path(args.output_dir)
        # Update all render nodes to use the new output directory
        for node in bpy.data.scenes["Scene"].node_tree.nodes:
            if hasattr(node, "base_path"):
                node.base_path = str(output_dir)
    
    # Generate the random light dataset
    generate_random_light_dataset(
        start_index=args.start_index,
        n_images=args.n_images,
        random_sphere=args.random_sphere,
        use_gpu=not args.no_gpu,
        show_progress=not args.no_progress
    )
    
    print(f"Generated {args.n_images} random light images starting from index {args.start_index}")


if __name__ == "__main__":
    main() 