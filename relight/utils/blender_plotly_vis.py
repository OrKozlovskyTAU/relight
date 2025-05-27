import bpy
import os
import sys
import logging
from mathutils import Vector, Matrix
import csv
from pathlib import Path

# Set up logging
logging.basicConfig(
    filename='blender_plotly_vis.log',
    filemode='w',
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

# Add the conda environment path (optional, as in the example)
site_packages = os.path.join('/home/dcor/orkozlovsky/miniconda3/envs/relight_blender/', 'lib', 'python3.10', 'site-packages')
if os.path.exists(site_packages):
    print(f"Adding {site_packages} to sys.path")
    sys.path.append(site_packages)

# Add relight source directory to PYTHONPATH
repo_root = '/home/dcor/orkozlovsky/repos/relight/'
os.environ['PYTHONPATH'] = repo_root + ':' + os.environ.get('PYTHONPATH', '')
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from relight.utils.blender_utils import orient_area_light_toward_bbox, get_area_light_size

import plotly.graph_objects as go
import numpy as np
import random


def _get_mesh_data(obj):
    """
    Extracts vertices and faces from a Blender mesh object in world coordinates.
    Returns:
        x, y, z: lists of vertex coordinates
        faces: list of (i, j, k) tuples for triangles
    """
    mesh = obj.data
    vertices = [obj.matrix_world @ v.co for v in mesh.vertices]
    x, y, z = zip(*[(v.x, v.y, v.z) for v in vertices])
    i, j, k = [], [], []
    for poly in mesh.polygons:
        verts = poly.vertices
        if len(verts) == 3:
            i.append(verts[0])
            j.append(verts[1])
            k.append(verts[2])
        elif len(verts) == 4:
            # Split quad into two triangles
            i.append(verts[0])
            j.append(verts[1])
            k.append(verts[2])
            i.append(verts[0])
            j.append(verts[2])
            k.append(verts[3])
        # Ngons are ignored for simplicity
    return x, y, z, i, j, k


def plot_blender_objects(object_names, colors=None, opacity=0.5, show=True):
    """
    Visualize multiple Blender mesh objects in a Plotly 3D figure.

    Args:
        object_names (list of str): Names of Blender mesh objects to visualize.
        colors (list of str, optional): List of Plotly color strings for each object.
        opacity (float): Opacity for all meshes.
        show (bool): Whether to immediately show the figure.

    Returns:
        plotly.graph_objs._figure.Figure: The Plotly 3D figure.
    """
    if colors is None:
        # Use Plotly's qualitative palette, repeat if needed
        palette = [
            '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
            '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
        ]
        colors = [palette[i % len(palette)] for i in range(len(object_names))]
    traces = []
    for idx, name in enumerate(object_names):
        obj = bpy.data.objects.get(name)
        if obj is None:
            print(f"Warning: Object '{name}' not found. Skipping.")
            continue
        if obj.type != 'MESH':
            print(f"Warning: Object '{name}' (type: '{obj.type}') is not a mesh. Skipping.")
            continue
        x, y, z, i, j, k = _get_mesh_data(obj)
        mesh3d = go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=opacity,
            color=colors[idx],
            name=name
        )
        traces.append(mesh3d)
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        title="Blender Scene Objects"
    )
    if show:
        fig.show()
    return fig


def plot_light_positions_with_scene(light_positions_dict, scene_object_names, show=False, output_html=None):
    """
    Visualize the scene geometry and the light source positions.
    Args:
        light_positions_dict: dict mapping (light_name, label) -> list of (x, y, z) positions
        scene_object_names: list of mesh object names to show as context
        show: whether to show the plot
        output_html: if given, save the plot to this HTML file
    """
    logger.info("Starting light positions visualization")
    logger.debug(f"Scene objects to plot: {scene_object_names}")
    logger.debug(f"Light positions dict contains {len(light_positions_dict)} entries")

    # Compute bbox_center from 'lights_bbox' object
    lights_bbox_obj = bpy.data.objects.get("lights_bbox")
    if lights_bbox_obj is None:
        logger.error("Blender object 'lights_bbox' not found in the scene. Area lights will not be oriented.")
        bbox_center = None
    else:
        bbox_corners = [lights_bbox_obj.matrix_world @ Vector(corner) for corner in lights_bbox_obj.bound_box]
        xs = [v.x for v in bbox_corners]
        ys = [v.y for v in bbox_corners]
        zs = [v.z for v in bbox_corners]
        bbox_center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2)
        logger.info(f"lights_bbox center: {bbox_center}")

    palette = [
        '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
        '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
    ]
    # Use plot_blender_objects to get the scene geometry as a base figure
    logger.debug("Creating base figure with scene objects")
    fig = plot_blender_objects(scene_object_names, colors=palette, opacity=0.3, show=False)
    
    # Add light positions
    logger.info("Adding light position traces to figure")
    for idx, ((light_name, label), positions) in enumerate(light_positions_dict.items()):
        logger.debug(f"Processing light '{light_name}' with label '{label}' ({len(positions)} positions)")
        obj = bpy.data.objects.get(light_name)
        if obj is None:
            logger.warning(f"Light object '{light_name}' not found in scene, skipping")
            continue
        if obj.type != 'LIGHT':
            logger.warning(f"Object '{light_name}' is not a light (type: {obj.type}), skipping")
            continue
            
        light_type = obj.data.type
        color = palette[(idx+len(scene_object_names)) % len(palette)]
        pos_arr = np.array(positions)
        if len(pos_arr) == 0:
            logger.warning(f"No positions to plot for light '{light_name}', skipping")
            continue
            
        trace_name = f"{label}/{light_name}"
        legendgroup = f"{label}/{light_name}"
        logger.debug(f"Creating trace for {trace_name} (type: {light_type})")
        
        if light_type == 'POINT':
            # Plot as dots
            logger.debug(f"Adding point light trace with {len(pos_arr)} positions")
            trace = go.Scatter3d(
                x=pos_arr[:,0], y=pos_arr[:,1], z=pos_arr[:,2],
                mode='markers',
                marker=dict(size=5, color=color),
                name=trace_name,
                legendgroup=legendgroup
            )
            fig.add_trace(trace)
        elif light_type == 'AREA':
            # Plot each position as a square (treat area light as square)
            size_x, _ = get_area_light_size(obj)
            size_y = size_x  # Assume square area light for visualization
            logger.debug(f"Adding area light traces for {len(positions)} positions (size: {size_x}x{size_y}, square assumed)")
            if bbox_center is not None:
                # Side effect: this changes the orientation of the light in the Blender scene
                # orient_area_light_toward_bbox(obj, bbox_center)
                logger.info(f"Oriented area light '{light_name}' to face bbox center {bbox_center}")
            for pos_idx, pos in enumerate(positions):
                local_corners = np.array([
                    [-size_x/2, -size_y/2, 0],
                    [ size_x/2, -size_y/2, 0],
                    [ size_x/2,  size_y/2, 0],
                    [-size_x/2,  size_y/2, 0],
                    [-size_x/2, -size_y/2, 0],  # close loop
                ])
                # Compute orientation as if the light were at 'pos' and facing bbox_center
                if bbox_center is not None:
                    direction = Vector(bbox_center) - Vector(pos)
                    direction.normalize()
                    rot = direction.to_track_quat('-Z', 'Y').to_matrix().to_3x3()
                else:
                    rot = Matrix.Identity(3)
                world_corners = np.array([rot @ Vector(corner) + Vector(pos) for corner in local_corners])
                logger.info(f"pos: {pos}, size_x: {size_x}, bbox_center: {bbox_center}")
                logger.info(f"direction: {direction}")
                logger.info(f"world_corners[0]: {world_corners[0]}")
                trace = go.Scatter3d(
                    x=world_corners[:,0], y=world_corners[:,1], z=world_corners[:,2],
                    mode='lines',
                    line=dict(color=color, width=4),
                    name=trace_name,
                    legendgroup=legendgroup,
                    showlegend=True if pos_idx == 0 else False  # Only show legend once per light
                )
                fig.add_trace(trace)

    logger.debug("Updating figure layout")
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
        ),
        title="Blender Scene with Light Source Positions"
    )
    
    if output_html:
        logger.info(f"Saving figure to {output_html}")
        fig.write_html(output_html)
    if show:
        logger.debug("Displaying figure")
        fig.show()
        
    logger.info("Light positions visualization complete")
    return fig


def plot_from_light_positions_csvs(
    base_dir,
    scene_object_names=None,
    show=False,
    output_html=None
):
    """
    Reads all light_positions*.csv files in base_dir and subdirectories,
    builds a plot_dict, and calls plot_light_positions_with_scene.

    Args:
        base_dir (str or Path): Directory to search for CSVs.
        scene_object_names (list): Blender mesh object names to plot as context.
        show (bool): Whether to show the plot.
        output_html (str or Path): If given, save the plot to this HTML file.
    """
    base_dir = Path(base_dir)
    if scene_object_names is None:
        scene_object_names = ["cornell_box", "large_box", "small_box"]

    plot_dict = {}

    for csv_path in base_dir.rglob("light_positions*.csv"):
        # Infer label from filename or parent directory
        if "val" in csv_path.parts:
            label = "val"
        elif "train" in csv_path.parts:
            if "subset" in csv_path.stem:
                subset_size = ''.join(filter(str.isdigit, csv_path.stem))
                label = f"subset{subset_size}"
            else:
                label = "train"
        else:
            label = csv_path.stem  # fallback

        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                light_name = row['light_name']
                x, y, z = float(row['x']), float(row['y']), float(row['z'])
                key = (light_name, label)
                plot_dict.setdefault(key, []).append((x, y, z))

    plot_light_positions_with_scene(
        plot_dict,
        scene_object_names,
        show=show,
        output_html=output_html
    )


# Example usage:
def main():
    """
    Example usage for visualizing light positions from CSVs:
    Change base_dir to your output directory containing light_positions*.csv files.
    """
    base_dir = "/home/dcor/orkozlovsky/repos/relight/data_v2"  # <-- Change this to your output directory
    scene_object_names = ["cornell_box", "large_box", "small_box"]
    output_html = "light_positions_plot.html"
    plot_from_light_positions_csvs(
        base_dir=base_dir,
        scene_object_names=scene_object_names,
        show=False,
        output_html=output_html
    )


if __name__ == "__main__":
    main()