import bpy
from mathutils import Vector

def print_light_bbox_ranges(eps=0):
    light_bbox_obj = bpy.data.objects.get("lights_bbox")
    if light_bbox_obj is None:
        print("Blender object 'lights_bbox' not found in the scene.")
        return
    bbox_corners = [light_bbox_obj.matrix_world @ Vector(corner) for corner in light_bbox_obj.bound_box]
    xs = [v.x for v in bbox_corners]
    ys = [v.y for v in bbox_corners]
    zs = [v.z for v in bbox_corners]
    x_range = [min(xs) + eps, max(xs) - eps]
    y_range = [min(ys) + eps, max(ys) - eps]
    z_range = [min(zs) + eps, max(zs) - eps]
    print(f"x_range: {x_range}")
    print(f"y_range: {y_range}")
    print(f"z_range: {z_range}")

if __name__ == "__main__":
    print("Objects in the scene:", [obj.name for obj in bpy.context.scene.objects])
    print_light_bbox_ranges() 