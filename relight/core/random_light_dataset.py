import bpy
import numpy as np
from mathutils import Vector
import math
import csv
from pathlib import Path

from relight.utils.blender_utils import (
    set_default_scene,
    setup_gpu_rendering
)


def inside_mesh(x, y, z, mesh):
    """Check if a point is inside a mesh."""
    p = Vector((x, y, z))
    max_dist = 1.0e20
    hit, point, normal, face = mesh.closest_point_on_mesh(p, distance=max_dist)
    p2 = point - p
    v = p2.dot(normal)
    return not (v < 0.0)


def generate_grid_positions(x_range, y_range, z_range, grid_size):
    """
    Generate a 3D grid of positions within the given bounds.
    
    Args:
        x_range: (min, max) for x coordinates
        y_range: (min, max) for y coordinates
        z_range: (min, max) for z coordinates
        grid_size: Number of points per dimension (will be rounded down to nearest power of 3)
        
    Returns:
        list: List of (x, y, z) positions
    """
    # Create grid points
    x_points = np.linspace(x_range[0], x_range[1], grid_size)
    y_points = np.linspace(y_range[0], y_range[1], grid_size)
    z_points = np.linspace(z_range[0], z_range[1], grid_size)
    
    # Generate all combinations
    positions = [(x, y, z) for x in x_points for y in y_points for z in z_points]
    
    return positions


def generate_random_light_dataset(start_index, n_images, output_dir, use_gpu=True, show_progress=True, grid_mode=False):
    """
    Generate a dataset of light positions in the scene.
    
    Args:
        start_index (int): The starting index for the dataset.
        n_images (int): The number of images to generate.
        use_gpu (bool): Whether to use GPU rendering if available.
        show_progress (bool): Whether to show progress information.
        grid_mode (bool): If True, use a 3D grid inside the Cornell box. If False, use random positions.
    """
    # Set up GPU rendering if requested
    if use_gpu:
        setup_gpu_rendering()
    
    # Set up the scene
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.data.objects["Area1"].hide_render = True
    bpy.data.objects["Area1"].hide_viewport = True
    # Enable point light
    bpy.data.objects["Point"].hide_render = False
    bpy.data.objects["Point"].hide_viewport = False
    
    # Get scene objects
    cornell_box = bpy.data.objects["cornell_box"]
    large_box = bpy.data.objects["large_box"]
    small_box = bpy.data.objects["small_box"]
    
    # Get render nodes
    render_png_node = bpy.data.scenes["Scene"].node_tree.nodes["render_png"]
    render_diffdir_png_node = bpy.data.scenes["Scene"].node_tree.nodes[
        "render_diffdir_png"
    ]
    render_diffindir_png_node = bpy.data.scenes["Scene"].node_tree.nodes[
        "render_diffindir_png"
    ]
   
    eps = 0.05
    # Define bounds for positioning based on the actual box dimensions
    x_range = [-0.25 + eps, 0.25 - eps]
    y_range = [-0.25 + eps, 0.4 - eps]
    z_range = [-0.25 + eps, 0.25 - eps]
    
    if show_progress:
        print(f"Using bounds: x={x_range}, y={y_range}, z={z_range}")
    
    # Generate light positions
    if grid_mode:
        # Calculate grid size based on n_images
        grid_size = int(math.ceil(n_images ** (1/3)))
        positions = generate_grid_positions(x_range, y_range, z_range, grid_size)
        
        # Filter out positions that are inside the mesh
        valid_positions = []
        for pos in positions:
            x, y, z = pos
            if not (inside_mesh(x, y, z, large_box) or inside_mesh(x, y, z, small_box)):
                valid_positions.append(pos)
        
        if show_progress:
            print(f"Using {grid_size}x{grid_size}x{grid_size} grid ({len(positions)} positions, {len(valid_positions)} valid)")
        
        # Adjust n_images based on available valid positions
        n_images = min(n_images, len(valid_positions))
        if n_images == 0:
            print("Error: No valid positions found in the grid. Try increasing the grid size.")
            return
        
        positions = valid_positions
    else:
        positions = None  # Will generate random positions in the loop
    
    # Create CSV file for light positions
    csv_path = output_dir / "light_positions.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['index', 'x', 'y', 'z'])
    
    count = start_index
    
    if show_progress:
        print(f"Generating {n_images} light images...")
    
    while count < start_index + n_images:
        # Position the light
        if grid_mode:
            # Use pre-generated grid positions
            x, y, z = positions[count - start_index]
        else:
            # Generate random position
            is_light_inside_mesh = True
            while is_light_inside_mesh:
                # Randomize point light location in reasonable bounds
                x = np.random.uniform(x_range[0], x_range[1])
                y = np.random.uniform(y_range[0], y_range[1])
                z = np.random.uniform(z_range[0], z_range[1])
                is_light_inside_mesh = inside_mesh(x, y, z, large_box) or inside_mesh(x, y, z, small_box)
        
        bpy.data.objects["Point"].location = (x, y, z)
        
        # Write light position to CSV
        with open(csv_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([count, x, y, z])
        
        # Set up file paths
        render_png_node.file_slots[0].path = f"{count:05d}_render_"
        render_diffdir_png_node.file_slots[0].path = f"{count:05d}_diffdir_"
        render_diffindir_png_node.file_slots[0].path = f"{count:05d}_diffindir_"
        
        # Render image, dirdiffuse, indirdiff, and light source as pngs
        bpy.ops.render.render()
        
        # Show progress
        if show_progress and (count - start_index + 1) % 10 == 0 or count == start_index + n_images - 1:
            print(f"Generated {count - start_index + 1}/{n_images} images ({(count - start_index + 1) / n_images * 100:.1f}%)")
        
        count += 1
    
    # Reset scene to default
    set_default_scene()
    
    if show_progress:
        print("Light dataset generation complete.")
        print(f"Light positions saved to {csv_path}") 