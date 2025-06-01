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
    def __init__(self, name, powers, mode):
        self.name = name
        self.powers = powers  # list of floats (W)
        self.mode = mode      # 'interior', 'faces', 'interior and faces'


def get_adjusted_ranges(light_obj, x_range, y_range, z_range):
    is_area = light_obj and light_obj.type == 'LIGHT' and light_obj.data.type == 'AREA'
    if is_area:
        size_x, _ = get_area_light_size(light_obj)
        size_y = size_x  # Assume area light is square for all calculations
        x_range_area = [x_range[0] + size_x/2, x_range[1] - size_x/2]
        y_range_area = [y_range[0] + size_y/2, y_range[1] - size_y/2]
        z_range_area = z_range  # Assuming area light is flat in XY, adjust if needed
        return x_range_area, y_range_area, z_range_area
    return x_range, y_range, z_range


def generate_light_grid_positions(light_obj, grid_size, mode, x_range, y_range, z_range, filter_valid_positions):
    # x_r, y_r, z_r = get_adjusted_ranges(light_obj, x_range, y_range, z_range) # TODO: uncomment this
    x_r, y_r, z_r = x_range, y_range, z_range
    x_points = np.linspace(x_r[0], x_r[1], grid_size)
    y_points = np.linspace(y_r[0], y_r[1], grid_size)
    z_points = np.linspace(z_r[0], z_r[1], grid_size)
    if mode == "interior":
        positions = [(x, y, z) for x in x_points for y in y_points for z in z_points]
        return filter_valid_positions(positions)
    elif mode == "faces":
        face_positions = []
        face_positions += [(x, y, z) for x in [x_r[0], x_r[1]] for y in y_points for z in z_points]
        face_positions += [(x, y, z) for y in [y_r[0], y_r[1]] for x in x_points for z in z_points]
        face_positions += [(x, y, z) for z in [z_r[0], z_r[1]] for x in x_points for y in y_points]
        face_positions = list({(round(x, 8), round(y, 8), round(z, 8)): (x, y, z) for (x, y, z) in face_positions}.values())
        return filter_valid_positions(face_positions)
    elif mode == "interior and faces":
        all_interior = generate_light_grid_positions(light_obj, grid_size, "interior", x_range, y_range, z_range, filter_valid_positions)
        all_faces = generate_light_grid_positions(light_obj, grid_size, "faces", x_range, y_range, z_range, filter_valid_positions)
        n = grid_size ** 3
        n_interior = n // 2
        n_faces = n - n_interior
        return all_interior[:n_interior] + all_faces[:n_faces]
    else:
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

    render_png_node = bpy.data.scenes["Scene"].node_tree.nodes["render_png"]
    render_diffdir_png_node = bpy.data.scenes["Scene"].node_tree.nodes["render_diffdir_png"]
    render_diffindir_png_node = bpy.data.scenes["Scene"].node_tree.nodes["render_diffindir_png"]

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

    def filter_valid_positions(positions):
        logger.debug(f"Filtering {len(positions)} positions for validity (not inside large or small box)...")
        filtered = [pos for pos in positions if not (inside_mesh(pos[0], pos[1], pos[2], large_box) or inside_mesh(pos[0], pos[1], pos[2], small_box))]
        logger.debug(f"Filtered down to {len(filtered)} valid positions.")
        return filtered

    # --- Visualization Data Structures ---
    vis_positions = {}
    pos_to_vis_pos = {}

    round_val = 3
    # --- TRAIN SET ---
    train_dir = output_dir / "train"
    os.makedirs(train_dir, exist_ok=True)
    render_png_node.base_path = str(train_dir)
    render_diffdir_png_node.base_path = str(train_dir)
    render_diffindir_png_node.base_path = str(train_dir)
    train_positions = []
    pos_to_index = {}
    logger.info("Starting TRAIN set generation loop.")
    for light_cfg in light_sources:
        logger.info(f"Processing TRAIN light source: {light_cfg.name} (mode={light_cfg.mode}, powers={light_cfg.powers})")
        light_obj = bpy.data.objects.get(light_cfg.name)
        train_positions_cfg = generate_light_grid_positions(light_obj, N, light_cfg.mode, x_range, y_range, z_range, filter_valid_positions)
        train_positions.extend([(round(x, round_val), round(y, round_val), round(z, round_val), light_cfg.name, light_cfg.powers) for (x, y, z) in train_positions_cfg])
    N_actual = len(train_positions)
    train_csv_path = train_dir / f"light_positions_{N_actual}.csv"
    logger.info(f"Writing train CSV to {train_csv_path} with {N_actual} positions.")
    with open(train_csv_path, 'w', newline='', buffering=1) as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['index', 'light_name', 'power', 'x', 'y', 'z'])
        count = 0
        for (x, y, z, name, powers) in train_positions:
            logger.debug(f"Setting light '{name}' to position ({x}, {y}, {z}) for TRAIN row {count}.")
            light_obj = bpy.data.objects.get(name)
            facing_point = get_facing_point((x, y, z), cornell_center, cornell_faces)
            logger.debug(f"[set_light_position_and_orientation] Light: {name}, Position: {(x, y, z)}, Facing point: {facing_point}")
            x_arr, y_arr, z_arr = set_light_position_and_orientation(light_obj, (x, y, z), facing_point)
            vis_positions.setdefault((name, light_obj.data.type, f"train_{N_actual}"), []).append((count, x_arr, y_arr, z_arr))
            pos_to_vis_pos[(name, x, y, z)] = (count, x_arr, y_arr, z_arr)
            light_obj.hide_render = False
            light_obj.hide_viewport = False
            for power in powers:
                logger.debug(f"Setting power {power} for light '{name}' at TRAIN row {count}.")
                light_obj.data.energy = power
                csv_writer.writerow([count, name, power, x, y, z])
                csvfile.flush()
                pos_to_index[(x, y, z, name, power)] = count
                render_png_node.file_slots[0].path = f"{count:05d}_render_"
                render_diffdir_png_node.file_slots[0].path = f"{count:05d}_diffdir_"
                render_diffindir_png_node.file_slots[0].path = f"{count:05d}_diffindir_"
                logger.debug(f"Rendering TRAIN image {count} for light '{name}' at position ({x}, {y}, {z}) with power {power}.")
                count += 1
                # bpy.ops.render.render() # TODO: uncomment this
            light_obj.hide_render = True
            light_obj.hide_viewport = True
    logger.info("Completed TRAIN set generation loop.")

    # --- SUBSET CSVs ---
    subset_size = N
    subset_positions = train_positions
    logger.info("Starting SUBSETS generation loop.")
    while subset_size > 1:
        subset_size = subset_size // 2
        if subset_size <= 1:
            break
        subset_positions = []
        for light_cfg in light_sources:
            logger.info(f"Processing SUBSET light source: {light_cfg.name} (mode={light_cfg.mode}, powers={light_cfg.powers}, subset_size={subset_size})")
            light_obj = bpy.data.objects.get(light_cfg.name)
            subset_positions_cfg = generate_light_grid_positions(light_obj, subset_size, light_cfg.mode, x_range, y_range, z_range, filter_valid_positions)
            subset_positions.extend([(round(x, round_val), round(y, round_val), round(z, round_val), light_cfg.name, light_obj.data.type, light_cfg.powers) for (x, y, z) in subset_positions_cfg])
        subset_N_actual = len(subset_positions)
        subset_csv_path = train_dir / f"light_positions_{subset_N_actual}.csv"
        logger.info(f"Writing SUBSET train CSV to {subset_csv_path} with {subset_N_actual} positions.")
        with open(subset_csv_path, 'w', newline='', buffering=1) as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['index', 'light_name', 'power', 'x', 'y', 'z'])
            # Map from position+name to original index in the full set
            for (x, y, z, name, light_type, powers) in subset_positions:
                orig_index, x_arr, y_arr, z_arr = pos_to_vis_pos[(name, x, y, z)]
                vis_positions.setdefault((name, light_type, f"train_{subset_N_actual}"), []).append((orig_index, x_arr, y_arr, z_arr))
                for power in powers:
                    orig_index = pos_to_index.get((x, y, z, name, power), None)
                    if orig_index is None:
                        logger.debug(f"Skipping SUBSET position ({x}, {y}, {z}, {name}) as it was not found in the original train set.")
                        continue
                    csv_writer.writerow([orig_index, name, power, x, y, z])
                    csvfile.flush()
    logger.info("Completed SUBSETS generation loop.")

    # --- VALIDATION SET ---
    val_dir = output_dir / "val"
    os.makedirs(val_dir, exist_ok=True)
    render_png_node.base_path = str(val_dir)
    render_diffdir_png_node.base_path = str(val_dir)
    render_diffindir_png_node.base_path = str(val_dir)
    val_csv_path = val_dir / "light_positions.csv"
    logger.info(f"Writing validation CSV to {val_csv_path}.")
    logger.info("Starting VAL set generation loop.")
    with open(val_csv_path, 'w', newline='', buffering=1) as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['index', 'light_name', 'power', 'x', 'y', 'z'])
        count = 0
        for light_cfg in light_sources:
            logger.info(f"Processing VAL light source: {light_cfg.name} (mode={light_cfg.mode}, powers={light_cfg.powers})")
            light_obj = bpy.data.objects.get(light_cfg.name)
            val_positions = generate_light_random_positions(light_obj, light_cfg.mode, Y, x_range, y_range, z_range, filter_valid_positions)
            logger.debug(f"{len(val_positions)} positions sampled for VAL light source {light_cfg.name}.")
            for i, (x, y, z) in enumerate(val_positions):
                logger.debug(f"Setting light '{light_cfg.name}' to position ({x}, {y}, {z}) for VAL row {count}.")
                facing_point = get_facing_point((x, y, z), cornell_center, cornell_faces)
                logger.debug(f"[set_light_position_and_orientation] Light: {light_cfg.name}, Position: {(x, y, z)}, Facing point: {facing_point}")
                x_arr, y_arr, z_arr = set_light_position_and_orientation(light_obj, (x, y, z), facing_point)
                vis_positions.setdefault((light_cfg.name, light_obj.data.type, "val"), []).append((count, x_arr, y_arr, z_arr))
                light_obj.hide_render = False
                light_obj.hide_viewport = False
                power = random.choice(light_cfg.powers)
                logger.debug(f"Setting power {power} for light '{light_cfg.name}' at VAL row {count}.")
                light_obj.data.energy = power
                csv_writer.writerow([count, light_cfg.name, power, x, y, z])
                csvfile.flush()
                render_png_node.file_slots[0].path = f"{count:05d}_render_"
                render_diffdir_png_node.file_slots[0].path = f"{count:05d}_diffdir_"
                render_diffindir_png_node.file_slots[0].path = f"{count:05d}_diffindir_"
                logger.debug(f"Rendering VAL image {count} for light '{light_cfg.name}' at position ({x}, {y}, {z}) with power {power}.")
                # bpy.ops.render.render() # TODO: uncomment this
            light_obj.hide_render = True
            light_obj.hide_viewport = True
    logger.info("Completed VAL set generation loop.")

    if show_progress:
        logger.info("Light dataset generation complete.")
        logger.info(f"Train positions saved to {train_csv_path}")
        logger.info(f"Validation positions saved to {val_csv_path}")

    # --- PLOTLY VISUALIZATION ---
    scene_object_names = ["cornell_box", "large_box", "small_box"]
    output_html = str(output_dir / "light_positions_plot.html")
    _tmp = {key: len(val) for key, val in vis_positions.items()}
    logger.info(f"vis_positions: {_tmp}")
    plot_light_positions_with_scene(vis_positions, scene_object_names, show=False, output_html=output_html)
    logger.info(f"Saved light positions plot to {output_html}") 