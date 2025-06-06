import bpy
import os
import sys
import logging
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


def get_palette(n):
    """
    Returns a color palette of length n, repeating Plotly's qualitative palette as needed.
    """
    palette = [
        '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
        '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
    ]
    return [palette[i % len(palette)] for i in range(n)]

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
        colors = get_palette(len(object_names))
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
        logger.info(f"Added mesh {name} with {len(x)} vertices and {len(i)} faces")
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
        light_positions_dict: dict mapping (light_name, light_type, label) -> list of (idx, x, y, z) positions
            - For POINT lights: x, y, z are values
            - For AREA lights: x, y, z are np.arrays representing the world_corners (shape: (N, 3))
        scene_object_names: list of mesh object names to show as context
        show: whether to show the plot
        output_html: if given, save the plot to this HTML file
    """
    logger.info("Starting light positions visualization")
    logger.debug(f"Scene objects to plot: {scene_object_names}")
    logger.debug(f"Light positions dict contains {len(light_positions_dict)} entries")

    palette = get_palette(len(scene_object_names) + len(light_positions_dict))

    # Use plot_blender_objects to get the scene geometry as a base figure
    logger.debug("Creating base figure with scene objects")
    fig = plot_blender_objects(scene_object_names, colors=palette, opacity=0.3, show=False)

    # Add light positions
    logger.info("Adding light position traces to figure")
    for idx, ((light_name, light_type, label), positions) in enumerate(light_positions_dict.items()):
        logger.debug(f"Processing light '{light_name}' (type: '{light_type}') with label '{label}' ({len(positions)} positions)")
        color = palette[(idx+len(scene_object_names)) % len(palette)]
        if len(positions) == 0:
            logger.warning(f"No positions to plot for light '{light_name}', skipping")
            continue
        indices = np.array([p[0] for p in positions])
        trace_name = f"{label}/{light_name}"
        legendgroup = f"{label}/{light_name}"
        if light_type == 'POINT':
            # Each position: (idx, x, y, z, powers)
            xs = np.array([p[1][0] for p in positions])
            ys = np.array([p[2][0] for p in positions])
            zs = np.array([p[3][0] for p in positions])
            idx_ranges = []
            powers_strs = []
            for p in positions:
                idx = p[0]
                powers = p[4]
                idx_range = f"{idx}-{idx+len(powers)-1}" if len(powers) > 1 else f"{idx}"
                idx_ranges.append(idx_range)
                powers_strs.append(", ".join(f"{pw:.2f}" for pw in powers))
            customdata = np.stack([idx_ranges, powers_strs], axis=1)
            trace = go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='markers',
                marker=dict(size=5, color=color),
                name=trace_name,
                legendgroup=legendgroup,
                customdata=customdata,
                hovertemplate='Light Index: %{customdata[0]}<br>Powers: %{customdata[1]} W<br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>'
            )
            fig.add_trace(trace)
        elif light_type == 'AREA':
            logger.debug(f"Adding area light traces for {len(positions)} positions (world_corners)")
            for i, (pos_idx, x, y, z, powers) in enumerate(positions):
                world_corners = np.stack([x, y, z], axis=1) if (isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and isinstance(z, np.ndarray)) else np.array([x, y, z]).T
                idx_range = f"{pos_idx}-{pos_idx+len(powers)-1}" if len(powers) > 1 else f"{pos_idx}"
                powers_str = ", ".join(f"{pw:.2f}" for pw in powers)
                customdata = np.full((world_corners.shape[0], 2), (idx_range, powers_str))
                trace = go.Scatter3d(
                    x=world_corners[:,0], y=world_corners[:,1], z=world_corners[:,2],
                    mode='lines',
                    line=dict(color=color, width=4),
                    name=trace_name,
                    legendgroup=legendgroup,
                    showlegend=i==0,
                    customdata=customdata,
                    hovertemplate='Light Index: %{customdata[0]}<br>Powers: %{customdata[1]} W<br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>'
                )
                fig.add_trace(trace)
        else:
            logger.warning(f"Unknown light type '{light_type}' for light '{light_name}', skipping")
            continue

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
            for idx, row in enumerate(reader):
                light_name = row['light_name']
                x, y, z = float(row['x']), float(row['y']), float(row['z'])
                # Use row["index"] if present, else fallback to idx
                row_index = int(row["index"]) if "index" in row and row["index"] != '' else idx
                key = (light_name, label)
                plot_dict.setdefault(key, []).append((row_index, x, y, z))

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