#!/usr/bin/env python
"""
Transport Matrix CLI

This script provides a command-line interface for generating and using transport matrices.
"""

import argparse
import bpy
from pathlib import Path

from relight.core.transport_matrix import (
    generate_transport_matrix,
    calculate_transport_matrix,
    save_transport_matrix,
    load_transport_matrix
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Transport Matrix CLI")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Generate transport matrix command
    generate_parser = subparsers.add_parser("generate", help="Generate transport matrix")
    generate_parser.add_argument("--proj-resx", type=int, default=64, help="Projector resolution in x direction")
    generate_parser.add_argument("--proj-resy", type=int, default=64, help="Projector resolution in y direction")
    generate_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing images")
    generate_parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    generate_parser.add_argument("--no-progress", action="store_true", help="Don't show progress")
    generate_parser.add_argument("--no-gpu", action="store_true", help="Don't use GPU rendering")
    
    # Calculate transport matrix command
    calculate_parser = subparsers.add_parser("calculate", help="Calculate transport matrix from rendered images")
    calculate_parser.add_argument("--proj-resx", type=int, default=64, help="Projector resolution in x direction")
    calculate_parser.add_argument("--proj-resy", type=int, default=64, help="Projector resolution in y direction")
    calculate_parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    calculate_parser.add_argument("--no-progress", action="store_true", help="Don't show progress")
    calculate_parser.add_argument("--multiprocessing", action="store_true", help="Use multiprocessing for faster computation")
    calculate_parser.add_argument("--output", type=str, help="Output file path for the transport matrix")
    
    # Load transport matrix command
    load_parser = subparsers.add_parser("load", help="Load a transport matrix from a file")
    load_parser.add_argument("--input", type=str, help="Input file path for the transport matrix")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    if args.command == "generate":
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
    
    elif args.command == "calculate":
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
    
    elif args.command == "load":
        # Load transport matrix
        input_path = args.input if args.input else None
        transport_matrix = load_transport_matrix(input_path)
        
        if transport_matrix is not None:
            print(f"Transport matrix loaded successfully. Shape: {transport_matrix.shape}")
        else:
            print("Failed to load transport matrix.")
    
    else:
        print("No command specified. Use --help for usage information.")


if __name__ == "__main__":
    main() 