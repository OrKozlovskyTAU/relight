import bpy
import numpy as np
import sys
from pathlib import Path
import os


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

from PIL import Image
import matplotlib.pyplot as plt
from relight.dataset.light_dataset import LightSourceConfiguration
from relight.utils.blender_utils import orient_area_light_toward_point, get_object_bbox_center_and_corners

# --- User argument: max irradiance ---
if '--' in sys.argv:
    idx = sys.argv.index('--')
    if idx + 1 < len(sys.argv):
        E = float(sys.argv[idx + 1])
    else:
        print("Usage: blender -b -P test_light_irradiance.py -- <max_irradiance>")
        sys.exit(1)
else:
    print("Usage: blender -b -P test_light_irradiance.py -- <max_irradiance>")
    sys.exit(1)

# --- Output path (as in light_dataset.py convention) ---
output_dir = Path("output/test_light_irradiance")
output_dir.mkdir(parents=True, exist_ok=True)

# --- Scene setup ---
scene = bpy.context.scene
scene.render.resolution_x = 512
scene.render.resolution_y = 512

# --- Get actual Cornell box center ---
cornell_box = bpy.data.objects["cornell_box"]
cornell_center, _ = get_object_bbox_center_and_corners(cornell_box)
cornell_center = np.array(cornell_center)

# --- Use existing lights ---
point_light = bpy.data.objects["Point"]
area_light = bpy.data.objects["Area1"]

test_pos = np.array([[0.0, -0.7, 0.0]])
point_light.location = test_pos[0]
area_light.location = test_pos[0]

# --- Irradiance values ---
irradiances = np.linspace(0, E, 10)

# --- Light configurations ---
point_cfg = LightSourceConfiguration('Point', irradiances.tolist(), 'interior')
area_cfg = LightSourceConfiguration('Area1', irradiances.tolist(), 'interior')

# --- Compute required powers (vectorized) ---
point_powers = point_cfg.compute_powers_vectorized(test_pos, cornell_center, point_light)[0]
area_powers = area_cfg.compute_powers_vectorized(test_pos, cornell_center, area_light)[0]

# --- Save image with histogram ---
def save_image_with_histogram(image_path, out_path, irradiance=None, power=None):
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    # Compute intensity (mean of RGB)
    intensity = img_np.mean(axis=2).flatten()
    # Plot histogram
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img)
    axs[0].axis('off')
    axs[0].set_title('Rendered Image')
    hist_title = 'Intensity Histogram'
    if irradiance is not None:
        hist_title += f' (Irrad: {irradiance:.2f} W/m², Power: {power:.2f} W)'
    axs[1].hist(intensity, bins=64, color='gray', range=(0, 255))
    axs[1].set_xlim(0, 255)
    axs[1].set_title(hist_title)
    axs[1].set_xlabel('Intensity')
    axs[1].set_ylabel('Pixel Count')
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def save_combined_figure(point_img_path, area_img_path, out_path, irradiance):
    # Load images
    point_img = Image.open(point_img_path).convert('RGB')
    area_img = Image.open(area_img_path).convert('RGB')
    point_np = np.array(point_img)
    area_np = np.array(area_img)
    # Compute intensities
    point_intensity = point_np.mean(axis=2).flatten()
    area_intensity = area_np.mean(axis=2).flatten()
    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs[0, 0].imshow(point_img)
    axs[0, 0].axis('off')
    axs[0, 0].set_title('Point Light Image')
    axs[0, 1].imshow(area_img)
    axs[0, 1].axis('off')
    axs[0, 1].set_title('Area Light Image')
    axs[1, 0].hist(point_intensity, bins=64, color='gray', range=(0, 255))
    axs[1, 0].set_xlim(0, 255)
    axs[1, 0].set_title('Point Light Histogram')
    axs[1, 0].set_xlabel('Intensity')
    axs[1, 0].set_ylabel('Pixel Count')
    axs[1, 1].hist(area_intensity, bins=64, color='gray', range=(0, 255))
    axs[1, 1].set_xlim(0, 255)
    axs[1, 1].set_title('Area Light Histogram')
    axs[1, 1].set_xlabel('Intensity')
    axs[1, 1].set_ylabel('Pixel Count')
    fig.suptitle(f'Irradiance: {irradiance:.2f} W/m²')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)

# --- Render for each irradiance ---
def render_both_lights(point_light, area_light, point_powers, area_powers, irradiances):
    for i, (point_power, area_power, irradiance) in enumerate(zip(point_powers, area_powers, irradiances)):
        # Render point light
        point_light.data.energy = point_power
        point_light.hide_render = False
        point_light.hide_viewport = False
        scene.render.filepath = str(output_dir / f'point_{i:02d}.png')
        bpy.ops.render.render(write_still=True)
        point_light.hide_render = True
        point_light.hide_viewport = True
        point_img_path = output_dir / f'point_{i:02d}.png'
        # Render area light
        area_light.data.energy = area_power
        area_light.hide_render = False
        area_light.hide_viewport = False
        orient_area_light_toward_point(area_light, cornell_center)
        scene.render.filepath = str(output_dir / f'area_{i:02d}.png')
        bpy.ops.render.render(write_still=True)
        area_light.hide_render = True
        area_light.hide_viewport = True
        area_img_path = output_dir / f'area_{i:02d}.png'
        # Save combined 2x2 figure
        combined_path = output_dir / f'combined_{i:02d}_Irrad_{irradiance:.2f}.png'
        save_combined_figure(point_img_path, area_img_path, combined_path, irradiance)

# Render both lights and save combined figures
render_both_lights(point_light, area_light, point_powers, area_powers, irradiances)

print(f"Combined figures saved to {output_dir}") 