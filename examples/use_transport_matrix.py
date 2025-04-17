#!/usr/bin/env python
"""
Example: Using the Transport Matrix

This script demonstrates how to use the transport matrix to relight a scene.
"""

import bpy
import numpy as np
from pathlib import Path
import argparse

# Add the project root to the Python path
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from relight.core.transport_matrix import load_transport_matrix
from relight.utils.blender_utils import (
    get_scene_resolution,
    load_texture,
    swap_projector_texture,
    setup_gpu_rendering
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Example: Using the Transport Matrix")
    parser.add_argument("--input", type=str, help="Input file path for the transport matrix")
    parser.add_argument("--target-image", type=str, help="Path to the target image to relight")
    parser.add_argument("--output", type=str, help="Output file path for the relit image")
    parser.add_argument("--proj-resx", type=int, default=64, help="Projector resolution in x direction")
    parser.add_argument("--proj-resy", type=int, default=64, help="Projector resolution in y direction")
    parser.add_argument("--no-gpu", action="store_true", help="Don't use GPU rendering")
    return parser.parse_args()


def relight_scene(transport_matrix, target_image_path, proj_resx=64, proj_resy=64, use_gpu=True):
    """
    Relight a scene using a transport matrix and a target image.
    
    Args:
        transport_matrix (numpy.ndarray): The transport matrix.
        target_image_path (str): Path to the target image.
        proj_resx (int): Resolution of the projector in the x direction.
        proj_resy (int): Resolution of the projector in the y direction.
        use_gpu (bool): Whether to use GPU rendering if available.
    
    Returns:
        numpy.ndarray: The relit image.
    """
    # Set up GPU rendering if requested
    if use_gpu:
        setup_gpu_rendering()
    
    # Get the scene resolution
    resx, resy = get_scene_resolution(bpy.context.scene)
    num_scene_pixels = resx * resy
    
    # Load the target image
    target_image = np.array(bpy.data.images.load(target_image_path).pixels).reshape(resy, resx, 4)
    target_image = target_image[:, :, :3]  # Remove alpha channel
    
    # Flatten the target image
    target_flat = target_image.reshape(-1, 3)
    
    # If the flattened image has fewer pixels than expected, pad with zeros
    if target_flat.shape[0] < num_scene_pixels:
        padding = np.zeros((num_scene_pixels - target_flat.shape[0], 3), dtype=np.float32)
        target_flat = np.vstack([target_flat, padding])
    
    # Truncate to the expected number of pixels
    target_flat = target_flat[:num_scene_pixels, :]
    
    # Solve for the projector image
    # This is a simple least squares solution
    # In practice, you might want to use a more sophisticated method
    projector_image = np.linalg.lstsq(transport_matrix.reshape(-1, 3), target_flat.reshape(-1, 3), rcond=None)[0]
    
    # Reshape the projector image
    projector_image = projector_image.reshape(proj_resy, proj_resx, 3)
    
    # Clamp the values to [0, 1]
    projector_image = np.clip(projector_image, 0, 1)
    
    # Create a texture from the projector image
    texture_key = "relight_texture"
    load_texture(projector_image, texture_key, proj_resy, proj_resx)
    swap_projector_texture(texture_key)
    
    # Render the scene
    bpy.ops.render.render()
    
    # Get the rendered image
    rendered_image = np.array(bpy.data.images["Viewer Node"].pixels).reshape(resy, resx, 4)
    rendered_image = rendered_image[:, :, :3]  # Remove alpha channel
    
    return rendered_image


def main():
    """Main function."""
    args = parse_args()
    
    # Load the transport matrix
    transport_matrix = load_transport_matrix(args.input)
    if transport_matrix is None:
        print("Failed to load transport matrix.")
        return
    
    # Relight the scene
    relit_image = relight_scene(
        transport_matrix,
        args.target_image,
        args.proj_resx,
        args.proj_resy,
        not args.no_gpu
    )
    
    # Save the relit image
    if args.output:
        output_path = args.output
    else:
        output_path = Path(args.target_image).parent / "relit.png"
    
    # Save the image using Blender's image API
    img = bpy.data.images.new("relit", width=relit_image.shape[1], height=relit_image.shape[0], alpha=False)
    img.pixels.foreach_set(relit_image.ravel())
    img.filepath_raw = str(output_path)
    img.save()
    
    print(f"Relit image saved to {output_path}")


if __name__ == "__main__":
    main() 