import csv
import json
import logging
import os
from math import pi
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from PIL import Image

from relight.utils.blender_plotly_vis import plot_light_positions_with_scene
from relight.utils.blender_utils import (
    get_area_light_size,
    get_cornell_faces,
    get_facing_point,
    get_object_bbox_center_and_corners,
    orient_area_light_toward_point,
    setup_gpu_rendering,
)

# Set up logging
logging.basicConfig(
    filename='light_dataset.log',
    filemode='w',
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.DEBUG,
    force=True
)
logger = logging.getLogger(__name__)


def save_config_to_json(config, output_dir):
    """Save configuration arguments to JSON file."""
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    logger.info(f"Configuration saved to {config_path}")


def load_config_from_json(output_dir):
    """Load configuration arguments from JSON file."""
    config_path = output_dir / "config.json"
    if not config_path.exists():
        return None
    with open(config_path, 'r') as f:
        config = json.load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config


def configs_match(config1, config2):
    """Check if two configurations match."""
    # Convert to strings for comparison to handle numpy arrays and other non-serializable types
    return json.dumps(config1, default=str, sort_keys=True) == json.dumps(config2, default=str, sort_keys=True)


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


class LightSourceConfiguration:
    def __init__(self, name, irradiances, mode, light_type=None):
        self.name = name
        self.irradiances = irradiances  # list of floats (W/m^2)
        self.mode = mode      # 'interior', 'faces', 'interior and faces'
        self.light_type = light_type  # Optional: 'POINT', 'AREA', etc.

    def compute_powers_vectorized(self, positions, target_point, light_obj):
        """
        Vectorized computation of required powers for all positions and all irradiances.
        positions: np.ndarray of shape (N, 3)
        target_point: (x, y, z)
        light_obj: Blender light object
        Returns: np.ndarray of shape (N, len(self.irradiances))
        """
        positions = np.asarray(positions)
        d = positions - np.array(target_point)[None, :]
        r2 = np.sum(d**2, axis=1)
        E = np.array(self.irradiances)  # shape (M,)
        if light_obj.data.type == 'POINT':
            # P = E * 4 * pi * r^2
            powers = E[None, :] * 4 * pi * r2[:, None]
        elif light_obj.data.type == 'AREA':
            # Use the exact solid angle formula for a rectangle, assuming the target is on the normal axis
            size_x, _ = get_area_light_size(light_obj)
            w, h = size_x, size_x  # assume square
            d = positions - np.array(target_point)[None, :]
            z = np.linalg.norm(d, axis=1)
            denom = 2 * z * np.sqrt(4 * z**2 + w**2 + h**2)
            omega = 4 * np.arctan2(w * h, denom)
            omega = np.clip(omega, 1e-8, None)
            powers = E[None, :] * w * h * pi / omega[:, None]
        else:
            # Default to point light formula
            powers = E[None, :] * 4 * pi * r2[:, None]
        return powers  # shape (N, M)


def get_adjusted_ranges(light_obj, x_range, y_range, z_range):
    is_area = light_obj and light_obj.type == 'LIGHT' and light_obj.data.type == 'AREA'
    if is_area:
        size_x, _ = get_area_light_size(light_obj)
        size_y = size_x  # Assume area light is square for all calculations
        x_range_area = [x_range[0] + size_x/2, x_range[1] - size_x/2]
        y_range_area = [y_range[0] + size_y/2, y_range[1] - size_y/2]
        z_range_area = [z_range[0] + size_x/2, z_range[1] - size_x/2]
        return x_range_area, y_range_area, z_range_area
    return x_range, y_range, z_range


def generate_light_grid_positions(light_obj, grid_size, mode, x_range, y_range, z_range, filter_valid_positions, eps, round_val, sampling_step=1):
    logger.debug(f"[generate_light_grid_positions] light_obj: {getattr(light_obj, 'name', None)}, grid_size: {grid_size}, mode: {mode}, x_range: {x_range}, y_range: {y_range}, z_range: {z_range}, eps: {eps}, round_val: {round_val}, sampling_step: {sampling_step}")
    x_r, y_r, z_r = get_adjusted_ranges(light_obj, x_range, y_range, z_range)
    x_points_full = np.linspace(x_r[0], x_r[1], grid_size)
    y_points_full = np.linspace(y_r[0], y_r[1], grid_size)
    z_points_full = np.linspace(z_r[0], z_r[1], grid_size)
    # Always include the last point in each axis
    def sample_axis(points):
        sampled = list(points[::sampling_step])
        if len(points) > 0 and points[-1] != sampled[-1]:
            sampled.append(points[-1])
        return sampled
    x_points = sample_axis(x_points_full)
    y_points = sample_axis(y_points_full)
    z_points = sample_axis(z_points_full)
    if mode == "interior":
        positions = [(x, y, z) for x in x_points for y in y_points for z in z_points]
        logger.debug(f"[generate_light_grid_positions] Generated {len(positions)} interior positions before filtering.")
        filtered = filter_valid_positions(positions)
        logger.debug(f"[generate_light_grid_positions] {len(filtered)} interior positions after filtering.")
        return filtered
    elif mode == "faces":
        face_positions = []
        face_positions += [(x, y, z) for x in [x_range[0] + eps, x_range[1] - eps] for y in y_points for z in z_points]
        face_positions += [(x, y, z) for y in [y_range[0] + eps, y_range[1] - eps] for x in x_points for z in z_points]
        face_positions += [(x, y, z) for z in [z_range[0] + eps, z_range[1] - eps] for x in x_points for y in y_points]
        face_positions = list({(round(x, round_val), round(y, round_val), round(z, round_val)): (x, y, z) for (x, y, z) in face_positions}.values())
        logger.debug(f"[generate_light_grid_positions] Generated {len(face_positions)} face positions before filtering.")
        filtered = filter_valid_positions(face_positions)
        logger.debug(f"[generate_light_grid_positions] {len(filtered)} face positions after filtering.")
        return filtered
    elif mode == "interior and faces":
        all_interior = generate_light_grid_positions(light_obj, grid_size, "interior", x_range, y_range, z_range, filter_valid_positions, eps, round_val, sampling_step)
        all_faces = generate_light_grid_positions(light_obj, grid_size, "faces", x_range, y_range, z_range, filter_valid_positions, eps, round_val, sampling_step)
        n = len(x_points) * len(y_points) * len(z_points)
        n_interior = n // 2
        n_faces = n - n_interior
        logger.debug(f"[generate_light_grid_positions] Combining {n_interior} interior and {n_faces} face positions.")
        return all_interior[:n_interior] + all_faces[:n_faces]
    else:
        logger.error(f"[generate_light_grid_positions] Unknown mode: {mode}")
        raise ValueError(f"Unknown mode: {mode}")


def generate_light_random_positions(light_obj, mode, n, x_range, y_range, z_range, filter_valid_positions):
    x_r, y_r, z_r = get_adjusted_ranges(light_obj, x_range, y_range, z_range)
    if mode == "interior":
        positions = []
        while len(positions) < n:
            x = np.random.uniform(x_r[0], x_r[1])
            y = np.random.uniform(y_r[0], y_r[1])
            z = np.random.uniform(z_r[0], z_r[1])
            positions.append((x, y, z))
        return filter_valid_positions(positions)[:n]
    elif mode == "faces":
        positions = []
        faces = [
            ("x", x_r[0]), ("x", x_r[1]),
            ("y", y_r[0]), ("y", y_r[1]),
            ("z", z_r[0]), ("z", z_r[1]),
        ]
        while len(positions) < n:
            face = faces[np.random.randint(0, 6)]
            if face[0] == "x":
                x = face[1]
                y = np.random.uniform(y_r[0], y_r[1])
                z = np.random.uniform(z_r[0], z_r[1])
            elif face[0] == "y":
                y = face[1]
                x = np.random.uniform(x_r[0], x_r[1])
                z = np.random.uniform(z_r[0], z_r[1])
            else:  # face[0] == "z"
                z = face[1]
                x = np.random.uniform(x_r[0], x_r[1])
                y = np.random.uniform(y_r[0], y_r[1])
            positions.append((x, y, z))
        return filter_valid_positions(positions)[:n]
    elif mode == "interior and faces":
        n_interior = n // 2
        n_faces = n - n_interior
        return generate_light_random_positions(light_obj, "interior", n_interior, x_range, y_range, z_range, filter_valid_positions) + \
               generate_light_random_positions(light_obj, "faces", n_faces, x_range, y_range, z_range, filter_valid_positions)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def set_light_position_and_orientation(light_obj, pos, facing_point):
    light_obj.location = pos
    is_area = light_obj and light_obj.type == 'LIGHT' and light_obj.data.type == 'AREA'
    if is_area:
        orient_area_light_toward_point(light_obj, facing_point)
        # Compute corners using the object's size and orientation (not bound_box)
        size_x, _ = get_area_light_size(light_obj)
        size_y = size_x  # Assume square
        local_corners = np.array([
            [-size_x/2, -size_y/2, 0],
            [ size_x/2, -size_y/2, 0],
            [ size_x/2,  size_y/2, 0],
            [-size_x/2,  size_y/2, 0],
            [-size_x/2, -size_y/2, 0],  # close loop
        ])
        corners = [light_obj.matrix_world @ Vector(corner) for corner in local_corners]
        x_arr = np.array([v.x for v in corners])
        y_arr = np.array([v.y for v in corners])
        z_arr = np.array([v.z for v in corners])
    else:
        # For non-area lights, just return the position as arrays of length 1
        x_arr = np.array([pos[0]])
        y_arr = np.array([pos[1]])
        z_arr = np.array([pos[2]])
    logger.debug(f"[set_light_position_and_orientation] Light: {light_obj.name}, Position: {(x_arr, y_arr, z_arr)}")
    return x_arr, y_arr, z_arr


def filter_valid_positions(positions, large_box, small_box):
    """Filter out positions inside large or small box."""
    filtered = [pos for pos in positions if not (inside_mesh(pos[0], pos[1], pos[2], large_box) or inside_mesh(pos[0], pos[1], pos[2], small_box))]
    logger.debug(f"Filtered {len(filtered)} valid positions out of {len(positions)}.")
    return filtered


def write_light_positions_csv(csv_path, positions):
    """
    Write light positions to CSV.
    positions: list of (index, x, y, z, name, powers)
    """
    with open(csv_path, 'w', newline='', buffering=1) as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['index', 'light_name', 'power', 'x', 'y', 'z'])
        for (indexes, x, y, z, name, powers) in positions:
            for index, power in zip(indexes, powers):
                csv_writer.writerow([index, name, power, x, y, z])
                csvfile.flush()


def load_light_positions_from_csv(csv_path):
    """
    Load light positions from CSV.
    Returns: list of (index, x, y, z, name, powers)
    """
    positions = []
    with open(csv_path, 'r', newline='') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        current_pos = None
        current_indexes = []
        current_powers = []
        
        for row in csv_reader:
            x, y, z = float(row['x']), float(row['y']), float(row['z'])
            name = row['light_name']
            power = float(row['power'])
            index = int(row['index'])
            
            if current_pos is None or current_pos != (x, y, z, name):
                # Save previous position if exists
                if current_pos is not None:
                    positions.append((current_indexes, current_pos[0], current_pos[1], current_pos[2], current_pos[3], current_powers))
                
                # Start new position
                current_pos = (x, y, z, name)
                current_indexes = [index]
                current_powers = [power]
            else:
                # Same position, add to current
                current_indexes.append(index)
                current_powers.append(power)
        
        # Add the last position
        if current_pos is not None:
            positions.append((current_indexes, current_pos[0], current_pos[1], current_pos[2], current_pos[3], current_powers))
    
    logger.info(f"Loaded {len(positions)} positions from {csv_path}")
    return positions


def render_light_positions(positions, cornell_center, cornell_faces, render_nodes, vis_positions, pos_to_vis_pos, pos_to_index, set_name, eps = 0, render=True):
    """
    Set light positions, render, and update vis_positions.
    positions: list of (index, x, y, z, name, powers)
    set_name: e.g. 'train_XXX', 'val', etc.
    """
    logger.info(f"[render_light_positions] Rendering {len(positions)} positions for set '{set_name}' with eps={eps}")
    
    # Get the base path from render nodes for file existence checking
    base_path = Path(render_nodes['render'].base_path)
    
    for (indexes, x, y, z, name, powers) in positions:
        light_obj = bpy.data.objects.get(name)
        facing_point = get_facing_point((x, y, z), cornell_center, cornell_faces)
        x_arr, y_arr, z_arr = set_light_position_and_orientation(light_obj, (x, y, z), facing_point)
        vis_positions.setdefault((name, light_obj.data.type, set_name), []).append((indexes, x_arr, y_arr, z_arr, powers))
        pos_to_vis_pos[(name, x, y, z)] = (indexes, x_arr, y_arr, z_arr, powers)
        light_obj.hide_render = False
        light_obj.hide_viewport = False
        for index, power in zip(indexes, powers):
            light_obj.data.energy = power
            pos_to_index[(x, y, z, name, power)] = index
            
            if render:
                try:
                    validate_and_render_files(index, render_nodes, base_path)
                    logger.debug(f"[render_light_positions] Successfully rendered/validated index {index} for light {name} at ({x}, {y}, {z}) with power {power}")
                except RuntimeError as e:
                    logger.error(f"[render_light_positions] Failed to render index {index} for light {name} at ({x}, {y}, {z}) with power {power}: {e}")
                    raise
        light_obj.hide_render = True
        light_obj.hide_viewport = True
    logger.info(f"[render_light_positions] Finished rendering set '{set_name}'")


def validate_and_render_files(index, render_nodes, base_path, max_retries=5):
    """
    Validate file sizes and render if necessary with retry logic.
    
    Args:
        index: The index for the file names
        render_nodes: Dictionary containing render nodes
        base_path: Base path for the output files
        max_retries: Maximum number of retry attempts
        
    Returns:
        bool: True if files are valid after rendering, False otherwise
        
    Raises:
        RuntimeError: If files are still invalid after max_retries attempts
    """
    # Expected file sizes in bytes
    EXPECTED_SIZES = {
        'render': 3152067,
        'diffdir': 4202354
    }
    
    def check_file_sizes():
        """Check if files exist and have correct sizes."""
        render_file = base_path / f"{index:05d}_render_0095.png"
        diffdir_file = base_path / f"{index:05d}_diffdir_0095.png"
        
        if not render_file.exists() or not diffdir_file.exists():
            return False
        
        logger.debug(f"[validate_and_render_files] Files exist for index {index}, checking sizes")
        
        try:
            render_size = render_file.stat().st_size
            diffdir_size = diffdir_file.stat().st_size
            
            if render_size != EXPECTED_SIZES['render']:
                logger.debug(f"[validate_and_render_files] Render file size mismatch for index {index}: expected {EXPECTED_SIZES['render']}, got {render_size}")
            if diffdir_size != EXPECTED_SIZES['diffdir']:
                logger.debug(f"[validate_and_render_files] Diffdir file size mismatch for index {index}: expected {EXPECTED_SIZES['diffdir']}, got {diffdir_size}")
                                
            return (render_size == EXPECTED_SIZES['render'] and diffdir_size == EXPECTED_SIZES['diffdir'])
        except OSError:
            return False
    
    def check_image_validity():
        """Check if images can be opened and converted to RGB."""
        render_file = base_path / f"{index:05d}_render_0095.png"
        diffdir_file = base_path / f"{index:05d}_diffdir_0095.png"
        
        try:
            # Try to open and convert both images to RGB
            Image.open(render_file).convert("RGB")
            Image.open(diffdir_file).convert("RGB")
            logger.debug(f"[validate_and_render_files] Images for index {index} are valid")
            return True
        except Exception as e:
            logger.debug(f"[validate_and_render_files] Image validation failed for index {index}: {e}")
            return False
    
    def render_files():
        """Render the files."""
        render_nodes['render'].file_slots[0].path = f"{index:05d}_render_"
        render_nodes['diffdir'].file_slots[0].path = f"{index:05d}_diffdir_"
        bpy.ops.render.render()
        logger.debug(f"[validate_and_render_files] Rendered index {index}")
    
    # Check if files already exist and are valid
    if check_file_sizes() and check_image_validity():
        logger.debug(f"[validate_and_render_files] Files for index {index} already exist and are valid, skipping render")
        return True
    
    # Try rendering with retry logic
    for attempt in range(max_retries):
        logger.debug(f"[validate_and_render_files] Attempt {attempt + 1}/{max_retries} for index {index}")
        
        # Render the files
        render_files()
        
        # Check if files are now valid
        if check_file_sizes() and check_image_validity():
            logger.debug(f"[validate_and_render_files] Files for index {index} are valid after attempt {attempt + 1}")
            return True
        
        logger.warning(f"[validate_and_render_files] Files for index {index} are invalid after attempt {attempt + 1}")
    
    # If we get here, all attempts failed
    error_msg = f"Failed to generate valid files for index {index} after {max_retries} attempts"
    logger.error(f"[validate_and_render_files] {error_msg}")
    raise RuntimeError(error_msg)


def generate_train_set(light_sources, N, x_range, y_range, z_range, filter_valid_fn, cornell_center, cornell_faces, render_nodes, train_dir, round_val, vis_positions, pos_to_vis_pos, pos_to_index, eps):
    logger.info(f"[generate_train_set] Generating train set with N={N}, eps={eps}")
    
    # 2. Check if train_csv_path exists
    train_positions = []
    N_actual = 0
    
    # Find the largest existing train CSV file
    existing_csv_files = list(train_dir.glob("light_positions_*.csv"))
    if existing_csv_files:
        # Sort by the number in filename to find the largest
        def extract_number(filename):
            try:
                return int(filename.stem.split('_')[-1])
            except (ValueError, IndexError):
                return 0
        
        largest_csv = max(existing_csv_files, key=extract_number)
        N_actual = extract_number(largest_csv)
        train_csv_path = largest_csv
        
        logger.info(f"[generate_train_set] Found existing train CSV: {train_csv_path} with {N_actual} positions")
        
        # 2.1 Load train_positions from existing CSV
        train_positions = load_light_positions_from_csv(train_csv_path)
        logger.info(f"[generate_train_set] Loaded {len(train_positions)} positions from existing CSV")
    
    # 2.2 Calculate train_positions if not loaded from CSV
    if not train_positions:
        logger.info(f"[generate_train_set] No existing CSV found, calculating new train positions")
        for light_cfg in light_sources:
            light_obj = bpy.data.objects.get(light_cfg.name)
            logger.debug(f"[generate_train_set] Generating grid positions for light '{light_cfg.name}'")
            train_positions_cfg = generate_light_grid_positions(light_obj, N, light_cfg.mode, x_range, y_range, z_range, filter_valid_fn, eps, round_val, sampling_step=1)
            positions_arr = np.array([(round(x, round_val), round(y, round_val), round(z, round_val)) for (x, y, z) in train_positions_cfg])
            powers_arr = light_cfg.compute_powers_vectorized(positions_arr, cornell_center, light_obj)  # shape (N, M)
            train_positions.extend([([len(powers)*index + i for i in range(len(powers))], pos[0], pos[1], pos[2], light_cfg.name, powers) for index, pos, powers in zip(range(len(train_positions), len(positions_arr) + len(train_positions)), positions_arr, powers_arr)])
        
        N_actual = len(train_positions)
        logger.info(f"[generate_train_set] Calculated {N_actual} new train positions")
        
        # Save to CSV
        train_csv_path = train_dir / f"light_positions_{N_actual}.csv"
        write_light_positions_csv(train_csv_path, train_positions)
        logger.info(f"[generate_train_set] Train positions saved to {train_csv_path}")
    
    # 3. Load train_positions from CSV and continue to render_light_positions
    # (This step is already done above, but we ensure we have the data)
    if not train_positions:
        logger.error("[generate_train_set] Failed to load or calculate train positions")
        raise RuntimeError("No train positions available")
    
    logger.info(f"[generate_train_set] Proceeding with {len(train_positions)} train positions")
    render_light_positions(train_positions, cornell_center, cornell_faces, render_nodes, vis_positions, pos_to_vis_pos, pos_to_index, f"train_{N_actual}", eps)
    logger.info(f"[generate_train_set] Train set generation complete with {N_actual} positions")


def generate_subsets(light_sources, N, x_range, y_range, z_range, filter_valid_fn, train_dir, round_val, vis_positions, pos_to_vis_pos, eps):
    """
    Generate subsets by subsampling the original train set grid using increasing sampling_step.
    Each subset is a uniform subsample of the original grid, not a new grid of smaller size.
    """
    logger.info(f"[generate_subsets] Generating subsets with initial N={N}, eps={eps}")
    sampling_step = 2
    while sampling_step < N:
        subset_positions = []
        for light_cfg in light_sources:
            light_obj = bpy.data.objects.get(light_cfg.name)
            logger.debug(f"[generate_subsets] Generating grid positions for light '{light_cfg.name}' with sampling_step={sampling_step}")
            subset_positions_cfg = generate_light_grid_positions(light_obj, N, light_cfg.mode, x_range, y_range, z_range, filter_valid_fn, eps, round_val, sampling_step)
            positions_arr = np.array([(round(x, round_val), round(y, round_val), round(z, round_val)) for (x, y, z) in subset_positions_cfg])
            subset_positions.extend([(pos_to_vis_pos[(light_cfg.name, pos[0], pos[1], pos[2])][0], pos[0], pos[1], pos[2], light_cfg.name, pos_to_vis_pos[(light_cfg.name, pos[0], pos[1], pos[2])][4]) for pos in positions_arr])
        subset_N_actual = len(subset_positions)
        logger.info(f"[generate_subsets] Subset size: {subset_N_actual}")
        subset_csv_path = train_dir / f"light_positions_{subset_N_actual}.csv"
        write_light_positions_csv(subset_csv_path, subset_positions)
        for (indexes, x, y, z, name, powers) in subset_positions:
            light_obj = bpy.data.objects.get(name)
            _, x_arr, y_arr, z_arr, _ = pos_to_vis_pos[(name, x, y, z)]
            vis_positions.setdefault((name, light_obj.data.type, f"train_{subset_N_actual}"), []).append((indexes, x_arr, y_arr, z_arr, powers))
        sampling_step *= 2


def generate_val_set(light_sources, Y, x_range, y_range, z_range, filter_valid_fn, cornell_center, cornell_faces, render_nodes, val_dir, round_val, vis_positions, pos_to_vis_pos, pos_to_index, eps):
    logger.info(f"[generate_val_set] Generating validation set with Y={Y}, eps={eps}")
    
    val_csv_path = val_dir / "light_positions.csv"
    
    # Check if validation CSV already exists
    if val_csv_path.exists():
        logger.info(f"[generate_val_set] Validation CSV already exists, loading from {val_csv_path}")
        val_positions = load_light_positions_from_csv(val_csv_path)
    else:
        logger.info(f"[generate_val_set] No existing validation CSV found, calculating new validation positions")
        val_positions = []
        for light_cfg in light_sources:
            light_obj = bpy.data.objects.get(light_cfg.name)
            logger.debug(f"[generate_val_set] Generating random positions for light '{light_cfg.name}'")
            val_positions_cfg = generate_light_random_positions(light_obj, light_cfg.mode, Y, x_range, y_range, z_range, filter_valid_fn)
            positions_arr = np.array([(round(x, round_val), round(y, round_val), round(z, round_val)) for (x, y, z) in val_positions_cfg])
            powers_arr = light_cfg.compute_powers_vectorized(positions_arr, cornell_center, light_obj)
            val_positions.extend([([len(powers)*idx + i for i in range(len(powers))], pos[0], pos[1], pos[2], light_cfg.name, powers) for idx, pos, powers in zip(range(len(val_positions), len(positions_arr) + len(val_positions)), positions_arr, powers_arr)])
        
        write_light_positions_csv(val_csv_path, val_positions)
        logger.info(f"[generate_val_set] Validation positions saved to {val_csv_path}")
    
    render_light_positions(val_positions, cornell_center, cornell_faces, render_nodes, vis_positions, pos_to_vis_pos, pos_to_index, "val")
    logger.info(f"[generate_val_set] Validation set generation complete with {len(val_positions)} positions")


def generate_light_dataset(N, Y, light_sources, output_dir, use_gpu=True, show_progress=True):
    """
    N: number of grid points per axis for train set
    Y: number of random validation images
    light_sources: list of LightSourceConfiguration
    output_dir: Path
    """
    # Convert output_dir to Path if it's a string
    output_dir = Path(output_dir)
    
    # Create current configuration
    current_config = {
            'N': N,
            'Y': Y,
            'light_sources': [{'name': ls.name, 'irradiances': ls.irradiances, 'mode': ls.mode, 'light_type': ls.light_type} for ls in light_sources],
            'use_gpu': use_gpu,
            'show_progress': show_progress
        }
    
    save_config_to_json(current_config, output_dir)
    
    # 1. Check if output_dir exists and handle configuration
    if output_dir.exists():
        # Load existing configuration
        existing_config = load_config_from_json(output_dir)
        if existing_config is not None:
            # Check if configurations match
            if not configs_match(existing_config, current_config):
                logger.error("Configuration mismatch detected!")
                logger.error(f"Existing config: {existing_config}")
                logger.error(f"Current config: {current_config}")
                raise ValueError("Input arguments do not match the existing configuration in the output directory. Please use a different output directory or remove the existing one.")
            else:
                logger.info("Configuration matches existing setup. Proceeding with recovery.")
    else:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save current configuration
        save_config_to_json(current_config, output_dir)
        logger.info("Created new output directory and saved configuration.")

    if use_gpu:
        setup_gpu_rendering()

    logger.info("Starting light dataset generation.")
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 1024

    cornell_box = bpy.data.objects["cornell_box"]
    large_box = bpy.data.objects["large_box"]
    small_box = bpy.data.objects["small_box"]

    render_nodes = {
        'render': bpy.data.scenes["Scene"].node_tree.nodes["render_png"],
        'diffdir': bpy.data.scenes["Scene"].node_tree.nodes["render_diffdir_png"],
        # 'diffindir': bpy.data.scenes["Scene"].node_tree.nodes["render_diffindir_png"]
    }

    # Check all light sources exist at the beginning and are of type 'LIGHT'
    missing_lights = [cfg.name for cfg in light_sources if bpy.data.objects.get(cfg.name) is None]
    if missing_lights:
        logger.error(f"Light sources not found in scene: {missing_lights}")
        raise ValueError(f"Light sources not found in scene: {missing_lights}")
    wrong_type_lights = [cfg.name for cfg in light_sources if bpy.data.objects.get(cfg.name) is not None and bpy.data.objects.get(cfg.name).type != 'LIGHT']
    if wrong_type_lights:
        logger.error(f"Light sources not of type 'LIGHT': {wrong_type_lights}")
        raise ValueError(f"Light sources not of type 'LIGHT': {wrong_type_lights}")

    # Get bounding box of 'lights_bbox' object
    lights_bbox_obj = bpy.data.objects["lights_bbox"]
    if lights_bbox_obj is None:
        logger.error("Blender object 'lights_bbox' not found in the scene.")
        raise ValueError("Blender object 'lights_bbox' not found in the scene.")
    _, bbox_corners = get_object_bbox_center_and_corners(lights_bbox_obj)
    xs = [v.x for v in bbox_corners]
    ys = [v.y for v in bbox_corners]
    zs = [v.z for v in bbox_corners]
    eps = 0.001
    x_range = [min(xs) + eps, max(xs) - eps]
    y_range = [min(ys) + eps, max(ys) - eps]
    z_range = [min(zs) + eps, max(zs) - eps]
    logger.info(f"light bbox range: {x_range}, {y_range}, {z_range}")

    # Get cornell_box bounding box and center
    cornell_center, cornell_corners = get_object_bbox_center_and_corners(cornell_box)
    cornell_faces = get_cornell_faces(cornell_corners)

    filter_valid_fn = lambda positions: filter_valid_positions(positions, large_box, small_box)

    # --- Visualization Data Structures ---
    vis_positions = {}
    pos_to_vis_pos = {}
    pos_to_index = {}
    round_val = 3

    # --- TRAIN SET ---
    train_dir = output_dir / "train"
    os.makedirs(train_dir, exist_ok=True)
    for node in render_nodes.values():
        node.base_path = str(train_dir)
    generate_train_set(
        light_sources, N, x_range, y_range, z_range, filter_valid_fn, cornell_center, cornell_faces, render_nodes, train_dir, round_val, vis_positions, pos_to_vis_pos, pos_to_index, eps
    )

    # --- SUBSET CSVs ---
    generate_subsets(
        light_sources, N, x_range, y_range, z_range, filter_valid_fn, train_dir, round_val, vis_positions, pos_to_vis_pos, eps
    )

    # --- VALIDATION SET ---
    val_dir = output_dir / "val"
    os.makedirs(val_dir, exist_ok=True)
    for node in render_nodes.values():
        node.base_path = str(val_dir)
    generate_val_set(
        light_sources, Y, x_range, y_range, z_range, filter_valid_fn, cornell_center, cornell_faces, render_nodes, val_dir, round_val, vis_positions, pos_to_vis_pos, pos_to_index, eps
    )

    if show_progress:
        logger.info("Light dataset generation complete.")

    # --- PLOTLY VISUALIZATION ---
    scene_object_names = ["cornell_box", "large_box", "small_box"]
    output_html = str(output_dir / "light_positions_plot.html")
    _tmp = {key: len(val) for key, val in vis_positions.items()}
    logger.info(f"vis_positions: {_tmp}")
    plot_light_positions_with_scene(vis_positions, scene_object_names, show=False, output_html=output_html)
    logger.info(f"Saved light positions plot to {output_html}") 
