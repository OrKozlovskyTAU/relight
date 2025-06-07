import bpy
import numpy as np
from mathutils import Vector
import csv
import os
import random
import logging

from relight.utils.blender_utils import (
    setup_gpu_rendering,
    get_area_light_size,
    orient_area_light_toward_point,
    get_object_bbox_center_and_corners,
    get_cornell_faces,
    get_facing_point,
)
from relight.utils.blender_plotly_vis import plot_light_positions_with_scene
from math import pi

# Set up logging
logging.basicConfig(
    filename='light_dataset.log',
    filemode='w',
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.DEBUG,
    force=True
)
logger = logging.getLogger(__name__)


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


def render_light_positions(positions, cornell_center, cornell_faces, render_nodes, vis_positions, pos_to_vis_pos, pos_to_index, set_name, eps = 0, render=True):
    """
    Set light positions, render, and update vis_positions.
    positions: list of (index, x, y, z, name, powers)
    set_name: e.g. 'train_XXX', 'val', etc.
    """
    logger.info(f"[render_light_positions] Rendering {len(positions)} positions for set '{set_name}' with eps={eps}")
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
            render_nodes['render'].file_slots[0].path = f"{index:05d}_render_" 
            render_nodes['diffdir'].file_slots[0].path = f"{index:05d}_diffdir_"
            render_nodes['diffindir'].file_slots[0].path = f"{index:05d}_diffindir_"
            if render:
                bpy.ops.render.render()
            logger.debug(f"[render_light_positions] Rendered index {index} for light {name} at ({x}, {y}, {z}) with power {power}")
        light_obj.hide_render = True
        light_obj.hide_viewport = True
    logger.info(f"[render_light_positions] Finished rendering set '{set_name}'")


def generate_train_set(light_sources, N, x_range, y_range, z_range, filter_valid_fn, cornell_center, cornell_faces, render_nodes, train_dir, round_val, vis_positions, pos_to_vis_pos, pos_to_index, eps):
    logger.info(f"[generate_train_set] Generating train set with N={N}, eps={eps}")
    train_positions = []
    for light_cfg in light_sources:
        light_obj = bpy.data.objects.get(light_cfg.name)
        logger.debug(f"[generate_train_set] Generating grid positions for light '{light_cfg.name}'")
        train_positions_cfg = generate_light_grid_positions(light_obj, N, light_cfg.mode, x_range, y_range, z_range, filter_valid_fn, eps, round_val, sampling_step=1)
        positions_arr = np.array([(round(x, round_val), round(y, round_val), round(z, round_val)) for (x, y, z) in train_positions_cfg])
        powers_arr = light_cfg.compute_powers_vectorized(positions_arr, cornell_center, light_obj)  # shape (N, M)
        train_positions.extend([([len(powers)*index + i for i in range(len(powers))], pos[0], pos[1], pos[2], light_cfg.name, powers) for index, pos, powers in zip(range(len(train_positions), len(positions_arr) + len(train_positions)), positions_arr, powers_arr)])
    N_actual = len(train_positions)
    logger.info(f"[generate_train_set] Total train positions: {N_actual}")
    train_csv_path = train_dir / f"light_positions_{N_actual}.csv"
    write_light_positions_csv(train_csv_path, train_positions)
    render_light_positions(train_positions, cornell_center, cornell_faces, render_nodes, vis_positions, pos_to_vis_pos, pos_to_index, f"train_{N_actual}", eps, render=False)
    logger.info(f"[generate_train_set] Train positions saved to {train_csv_path}")


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
    val_positions = []
    for light_cfg in light_sources:
        light_obj = bpy.data.objects.get(light_cfg.name)
        logger.debug(f"[generate_val_set] Generating random positions for light '{light_cfg.name}'")
        val_positions_cfg = generate_light_random_positions(light_obj, light_cfg.mode, Y, x_range, y_range, z_range, filter_valid_fn)
        positions_arr = np.array([(round(x, round_val), round(y, round_val), round(z, round_val)) for (x, y, z) in val_positions_cfg])
        powers_arr = light_cfg.compute_powers_vectorized(positions_arr, cornell_center, light_obj)
        val_positions.extend([([len(powers)*idx + i for i in range(len(powers))], pos[0], pos[1], pos[2], light_cfg.name, powers) for idx, pos, powers in zip(range(len(val_positions), len(positions_arr) + len(val_positions)), positions_arr, powers_arr)])
    val_csv_path = val_dir / "light_positions.csv"
    write_light_positions_csv(val_csv_path, val_positions)
    render_light_positions(val_positions, cornell_center, cornell_faces, render_nodes, vis_positions, pos_to_vis_pos, pos_to_index, "val")
    logger.info(f"[generate_val_set] Validation positions saved to {val_csv_path}")


def generate_light_dataset(N, Y, light_sources, output_dir, use_gpu=True, show_progress=True):
    """
    N: number of grid points per axis for train set
    Y: number of random validation images
    light_sources: list of LightSourceConfiguration
    output_dir: Path
    """
    if use_gpu:
        setup_gpu_rendering()

    logger.info("Starting light dataset generation.")
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512

    cornell_box = bpy.data.objects["cornell_box"]
    large_box = bpy.data.objects["large_box"]
    small_box = bpy.data.objects["small_box"]

    render_nodes = {
        'render': bpy.data.scenes["Scene"].node_tree.nodes["render_png"],
        'diffdir': bpy.data.scenes["Scene"].node_tree.nodes["render_diffdir_png"],
        'diffindir': bpy.data.scenes["Scene"].node_tree.nodes["render_diffindir_png"]
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