import logging
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector
from PIL import Image


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


def get_area_light_size(light_obj):
    # Returns (size_x, size_y) for area light
    size_x = getattr(light_obj.data, 'size', 0.0)
    size_y = getattr(light_obj.data, 'size_y', size_x)
    return size_x, size_y


def orient_area_light_toward_point(light_obj, facing_point, set_matrix_world=True):
    # Orient the area light so its normal points toward facing_point
    direction = Vector(facing_point) - light_obj.location
    direction.normalize()
    # Default area light normal is (0, 0, -1) in local space
    Vector((0, 1, 0))
    normal = direction
    # Compute rotation matrix to align -Z to normal
    rot = normal.to_track_quat('-Z', 'Y').to_matrix().to_4x4()
    if set_matrix_world:
        light_obj.matrix_world = Matrix.Translation(light_obj.location) @ rot
    # Debug logging
    logger = logging.getLogger(__name__)
    logger.debug(f"[orient_area_light_toward_point] Light: {getattr(light_obj, 'name', None)}, Location: {tuple(light_obj.location)}, Facing point: {facing_point}, Direction: {tuple(direction)}")
    if not set_matrix_world:
        return rot.to_3x3()


def get_object_bbox_center_and_corners(obj):
    """
    Returns the center and corners of a Blender object's bounding box in world coordinates.
    """
    if obj is None:
        return None, None
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    xs = [v.x for v in corners]
    ys = [v.y for v in corners]
    zs = [v.z for v in corners]
    center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2)
    return center, corners


def get_cornell_faces(cornell_corners):
    """
    Returns face definitions for the Cornell box given its corners.
    """
    xs = [v.x for v in cornell_corners]
    ys = [v.y for v in cornell_corners]
    zs = [v.z for v in cornell_corners]
    faces = [
        ("x", round(min(xs), 2), (1, 0, 0)), ("x", round(max(xs), 2), (-1, 0, 0)),
        ("y", round(min(ys), 2), (0, 1, 0)), ("y", round(max(ys), 2), (0, -1, 0)),
        ("z", round(min(zs), 2), (0, 0, 1)), ("z", round(max(zs), 2), (0, 0, -1)),
    ]
    return faces

def is_in_bounds(pos, faces_values, eps=0.05):
    """
    Checks if a position is within the bounds of the faces.
    """
    logger = logging.getLogger(__name__)
    for pos_val, min_val, max_val in zip(pos, faces_values[:, 0], faces_values[:, 1]):
        if not min_val - eps <= pos_val <= max_val + eps:
            logger.debug(f"[is_in_bounds/vis] Position value: {pos_val}, Min value: {min_val}, Max value: {max_val}, Out of bounds")
            return False
    logger.debug(f"[is_in_bounds/vis] Position value: {pos_val}, Min value: {min_val}, Max value: {max_val}, In bounds")
    return True

def is_on_face(pos, axis, face_val, eps=0.05):
    """
    Checks if a position is on a face and within the bounds of that face.
    """
    logger = logging.getLogger(__name__)
    pos_val = pos[{"x": 0, "y": 1, "z": 2}[axis]]
    on_face = abs(pos_val - face_val) <= eps
    logger.debug(f"[is_on_face/vis] Position value: {pos_val}, Axis: {axis}, Face value: {face_val}, Diff: {round(abs(pos_val - face_val), 3)}, Eps: {eps}, On face: {on_face}")
    return on_face


def compute_rotation_from_vector(vector):
    """
    Compute a rotation matrix that orients the Z axis along the given vector (direction or normal).
    """
    vector = Vector(vector).normalized()
    up = Vector((0, 1, 0))
    if abs(vector.dot(up)) > 0.99:
        up = Vector((1, 0, 0))
    right = up.cross(vector).normalized()
    up = vector.cross(right).normalized()
    rot = Matrix((right, up, vector)).transposed()
    return rot


def get_facing_point(pos, cornell_center, cornell_faces, eps=0.05):
    """
    Determine the facing point for a light position:
    - If on a face (and in bounds), return a point along the face normal (pos + normal)
    - Otherwise, return the cornell_center
    """
    logger = logging.getLogger(__name__)
    if cornell_faces is not None:
        faces_values = np.array([face[1] for face in cornell_faces]).reshape(-1, 2)
        if is_in_bounds(pos, faces_values, eps):
            for axis, face_val, normal in cornell_faces:
                if is_on_face(pos, axis, face_val, eps):
                    logger.debug(f"[get_facing_point/vis] Position {pos} is on face {axis}={face_val} (normal={normal}), using pos+normal as facing_point.")
                    return tuple(np.array(pos) + np.array(normal))
    logger.debug(f"[get_facing_point/vis] Position {pos} is not on a face and in bounds, using cornell_center {cornell_center} as facing_point.")
    return tuple(cornell_center)
