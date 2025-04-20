import bpy
from pathlib import Path
import numpy as np
from PIL import Image
from mathutils import Vector


def get_active_object():
    """Get the currently active object in the Blender scene."""
    return bpy.context.view_layer.objects.active


def clean_modifiers(object):
    """Remove all modifiers from an object."""
    object.modifiers.clear()


def clean_nodegroups():
    """Remove all node groups from the Blender data."""
    for i in range(len(bpy.data.node_groups)):
        bpy.data.node_groups.remove(bpy.data.node_groups[0])


def set_default_scene():
    """Set up the default scene configuration."""
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 1024
    bpy.data.objects["Point"].hide_render = True
    bpy.data.objects["Point"].hide_viewport = True
    bpy.data.objects["Area1"].hide_render = False
    bpy.data.objects["Area1"].hide_viewport = False

    render_png_node = bpy.data.scenes["Scene"].node_tree.nodes["render_png"]
    render_png_node.base_path = str(Path(bpy.path.abspath("//"), "images"))
    render_png_node.file_slots[0].path = "render"

    render_diffdir_png_node = bpy.data.scenes["Scene"].node_tree.nodes[
        "render_diffdir_png"
    ]
    render_diffdir_png_node.base_path = str(Path(bpy.path.abspath("//"), "images"))
    render_diffdir_png_node.file_slots[0].path = "diffdir"

    render_diffindir_png_node = bpy.data.scenes["Scene"].node_tree.nodes[
        "render_diffindir_png"
    ]
    render_diffindir_png_node.base_path = str(Path(bpy.path.abspath("//"), "images"))
    render_diffindir_png_node.file_slots[0].path = "diffindir"

    render_diffcol_png_node = bpy.data.scenes["Scene"].node_tree.nodes[
        "render_diffcol_png"
    ]
    render_diffcol_png_node.base_path = str(Path(bpy.path.abspath("//"), "images"))
    render_diffcol_png_node.file_slots[0].path = "diffcol"
    
    clean_nodegroups()


def get_scene_resolution(scene):
    """Get the resolution of the scene, accounting for resolution percentage."""
    resolution_scale = scene.render.resolution_percentage / 100.0
    resolution_x = scene.render.resolution_x * resolution_scale  # [pixels]
    resolution_y = scene.render.resolution_y * resolution_scale  # [pixels]
    return int(resolution_x), int(resolution_y)


def remove_textures():
    """Remove all textures except for exceptions."""
    # remove current textures
    exceptions = ["marble.jpg"]
    string_exceptions = ["Wall"]
    for image in bpy.data.images:
        if image.name not in exceptions:
            if ~np.any(np.array([x in image.name for x in string_exceptions])):
                print("removing texture: {}".format(image))
                bpy.data.images.remove(image)


def load_texture(texture_path, texture_key, proj_h, proj_w, overwrite=True):
    """Load a texture from a file and create a Blender image."""
    for image in bpy.data.images:
        if overwrite and image.name == texture_key:
            bpy.data.images.remove(image)
    pilImage = Image.open(str(texture_path)).convert("RGB")
    image = np.asarray(pilImage)
    if image.shape[0] != proj_h or image.shape[1] != proj_w:
        pilImage = pilImage.resize((proj_w, proj_h))
        image = np.asarray(pilImage)
    float_texture = (image / 255).astype(np.float32)
    padded_texture = np.concatenate(
        (float_texture, np.ones_like(float_texture)[:, :, 0:1]), axis=-1
    )
    bpy_image = bpy.data.images.new(
        texture_key, width=proj_w, height=proj_h, alpha=False
    )
    bpy_image.pixels.foreach_set(padded_texture.ravel())


def save_texture(texture_key, proj_width, proj_height, dst):
    """Save a texture to a file."""
    # save current texture
    if dst.is_file():
        return
    print("saving to: {}".format(str(dst)))
    image = np.array(bpy.data.images[texture_key].pixels).reshape(
        proj_height, proj_width, 4
    )
    image = Image.fromarray((image[:, :, :3] * 255).astype(np.uint8))
    image.save(dst)


def swap_projector_texture(texture_name):
    """Swap the texture of the projector."""
    projector_name = "Projector"
    bpy.data.images[texture_name].colorspace_settings.name = "Linear"
    bpy.data.images[texture_name].source = "FILE"
    bpy.data.images[texture_name].filepath = str(
        Path(bpy.path.abspath("//"), "images", texture_name + ".png")
    )
    bpy.data.lights[projector_name].node_tree.nodes["Image Texture"].image = (
        bpy.data.images[texture_name]
    )


def hide_object_and_children(obj, hide=True):
    """Hide an object and all its children."""
    # hide the children
    obj.hide_viewport = hide
    obj.hide_render = hide
    for child in obj.children:
        child.hide_viewport = hide
        child.hide_render = hide


def setup_gpu_rendering():
    """Set up GPU rendering if available."""
    # Check if GPU is available
    prefs = bpy.context.preferences.addons['cycles'].preferences
    has_gpu = False
    for device in prefs.devices:
        if device.type == 'CUDA' or device.type == 'OPENCL':
            has_gpu = True
            device.use = True
    
    if has_gpu:
        print("GPU rendering is enabled.")
        return True
    else:
        print("No GPU devices found. Using CPU rendering.")
        return False 