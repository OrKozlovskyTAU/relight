#!/usr/bin/env python
"""
Light Dataset CLI

This script provides a command-line interface for generating light datasets (train and validation) with configurable light sources.
"""

import os
import argparse
from pathlib import Path
import sys

# Add relight source directory to PYTHONPATH
repo_root = '/home/dcor/orkozlovsky/repos/relight/'
os.environ['PYTHONPATH'] = repo_root + ':' + os.environ.get('PYTHONPATH', '')
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Add the conda environment path (optional, as in the example)
site_packages = os.path.join('/home/dcor/orkozlovsky/miniconda3/envs/relight_blender/', 'lib', 'python3.10', 'site-packages')
if os.path.exists(site_packages):
    print(f"Adding {site_packages} to sys.path")
    sys.path.append(site_packages)

from relight.dataset.light_dataset import generate_light_dataset, LightSourceConfiguration

def parse_args():
    """Parse command line arguments."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Light Dataset CLI")
    parser.add_argument("--N", type=int, default=10, help="Number of grid points per axis for the train set (N x N x N)")
    parser.add_argument("--Y", type=int, default=20, help="Number of random validation images")
    parser.add_argument("--no-gpu", action="store_true", help="Don't use GPU rendering")
    parser.add_argument("--no-progress", action="store_true", help="Don't show progress bar")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for the generated images")
    # Optionally: add arguments for light source config, or use a hardcoded example
    return parser.parse_args(argv)

def main():
    """Main function."""
    args = parse_args()
    output_dir = Path(args.output_dir)

    # Example: define light sources (customize as needed)
    light_sources = [
        LightSourceConfiguration(
            name="Point",
            powers=[1, 1.5, 2],
            mode="interior",
        ),
        LightSourceConfiguration(
            name="Area1",
            powers=[1, 1.5, 2],
            mode="faces",
        )
    ]

    # Generate the light dataset
    generate_light_dataset(
        N=args.N,
        Y=args.Y,
        light_sources=light_sources,
        output_dir=output_dir,
        use_gpu=not args.no_gpu,
        show_progress=not args.no_progress
    )

    print(f"Generated train and validation sets with N={args.N}, Y={args.Y}")

if __name__ == "__main__":
    main() 