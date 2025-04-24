#!/usr/bin/env python
"""
Relight - Main Script

This script provides a command-line interface for the Relight project.
"""

import argparse
import bpy
import sys
import os
from pathlib import Path


# Add the conda environment path
site_packages = os.path.join('/home/dcor/orkozlovsky/miniconda3/envs/relight_blender/', 'lib', 'python3.10', 'site-packages')
if os.path.exists(site_packages):
    print(f"Adding {site_packages} to sys.path")
    sys.path.append(site_packages)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from relight.core.transport_matrix import (
    generate_transport_matrix,
    calculate_transport_matrix,
    save_transport_matrix,
    load_transport_matrix
)
from relight.core.random_light_dataset import generate_random_light_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Relight - A Blender-based relighting tool")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Transport matrix commands
    transport_parser = subparsers.add_parser("transport", help="Transport matrix operations")
    transport_subparsers = transport_parser.add_subparsers(dest="transport_command", help="Transport matrix command")
    
    # Generate transport matrix command
    generate_parser = transport_subparsers.add_parser("generate", help="Generate transport matrix")
    generate_parser.add_argument("--proj-resx", type=int, default=64, help="Projector resolution in x direction")
    generate_parser.add_argument("--proj-resy", type=int, default=64, help="Projector resolution in y direction")
    generate_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing images")
    generate_parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    generate_parser.add_argument("--no-progress", action="store_true", help="Don't show progress")
    generate_parser.add_argument("--no-gpu", action="store_true", help="Don't use GPU rendering")
    
    # Calculate transport matrix command
    calculate_parser = transport_subparsers.add_parser("calculate", help="Calculate transport matrix from rendered images")
    calculate_parser.add_argument("--proj-resx", type=int, default=64, help="Projector resolution in x direction")
    calculate_parser.add_argument("--proj-resy", type=int, default=64, help="Projector resolution in y direction")
    calculate_parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    calculate_parser.add_argument("--no-progress", action="store_true", help="Don't show progress")
    calculate_parser.add_argument("--multiprocessing", action="store_true", help="Use multiprocessing for faster computation")
    calculate_parser.add_argument("--output", type=str, help="Output file path for the transport matrix")
    
    # Load transport matrix command
    load_parser = transport_subparsers.add_parser("load", help="Load a transport matrix from a file")
    load_parser.add_argument("--input", type=str, help="Input file path for the transport matrix")
    
    # Random light dataset commands
    dataset_parser = subparsers.add_parser("dataset", help="Random light dataset operations")
    dataset_parser.add_argument("--start-index", type=int, default=0, help="Starting index for the dataset")
    dataset_parser.add_argument("--n-images", type=int, default=100, help="Number of images to generate")
    dataset_parser.add_argument("--no-gpu", action="store_true", help="Don't use GPU rendering")
    dataset_parser.add_argument("--no-progress", action="store_true", help="Don't show progress")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    if args.command == "transport":
        if args.transport_command == "generate":
            # Generate transport matrix
            success = generate_transport_matrix(
                proj_resx=args.proj_resx,
                proj_resy=args.proj_resy,
                overwrite=args.overwrite,
                batch_size=args.batch_size,
                show_progress=not args.no_progress,
                use_gpu=not args.no_gpu
            )
            
            if success:
                print("Transport matrix generation completed successfully.")
            else:
                print("Transport matrix generation failed.")
        
        elif args.transport_command == "calculate":
            # Calculate transport matrix
            transport_matrix = calculate_transport_matrix(
                proj_resx=args.proj_resx,
                proj_resy=args.proj_resy,
                batch_size=args.batch_size,
                show_progress=not args.no_progress,
                use_multiprocessing=args.multiprocessing
            )
            
            if transport_matrix is not None:
                # Save the transport matrix
                output_path = args.output if args.output else None
                save_transport_matrix(transport_matrix, output_path)
                print("Transport matrix calculation completed successfully.")
            else:
                print("Transport matrix calculation failed.")
        
        elif args.transport_command == "load":
            # Load transport matrix
            input_path = args.input if args.input else None
            transport_matrix = load_transport_matrix(input_path)
            
            if transport_matrix is not None:
                print(f"Transport matrix loaded successfully. Shape: {transport_matrix.shape}")
            else:
                print("Failed to load transport matrix.")
        
        else:
            print("No transport command specified. Use --help for usage information.")
    
    elif args.command == "dataset":
        # Generate random light dataset
        generate_random_light_dataset(
            start_index=args.start_index,
            n_images=args.n_images,
            use_gpu=not args.no_gpu,
            show_progress=not args.no_progress
        )
        
        print("Random light dataset generation completed successfully.")
    
    else:
        print("No command specified. Use --help for usage information.")


if __name__ == "__main__":
    main() 