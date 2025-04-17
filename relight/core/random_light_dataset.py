import bpy
import numpy as np
from mathutils import Vector

from relight.utils.blender_utils import (
    set_default_scene,
    setup_gpu_rendering
)


def create_node(node_tree, type_name, node_x_location, node_location_step_x=0):
    """
    Creates a node of a given type, and sets/updates the location of the node on the X axis.
    Returning the node object and the next location on the X axis for the next node.
    """
    node_obj = node_tree.nodes.new(type=type_name)
    node_obj.location.x = node_x_location
    node_x_location += node_location_step_x

    return node_obj, node_x_location


def link_nodes_by_socket(node_tree, from_node, from_socket, to_node, to_socket):
    """Link two nodes by their sockets."""
    node_tree.links.new(from_node.outputs[from_socket], to_node.inputs[to_socket])


def set_intersection_target(source_object, target_object):
    """Set the target object for intersection testing."""
    # this function assumes create_intersection_nodegroup was called on source_object
    modifier = source_object.modifiers.get("Geometry Nodes")
    target_info_node = modifier.node_group.nodes["Object Info"]
    target_info_node.inputs[0].default_value = target_object


def create_intersection_nodegroup(source_object, target_object):
    """Create a geometry node group for intersection testing."""
    # clear modifiers and all node groups
    source_object.modifiers.clear()
    for i in range(len(bpy.data.node_groups)):
        bpy.data.node_groups.remove(bpy.data.node_groups[0])
    
    # create geo node modifier
    modifier = source_object.modifiers.new(type="NODES", name="Geometry Nodes")
    
    # Set the active object to the source_object to ensure proper context
    bpy.context.view_layer.objects.active = source_object
    
    # create a new geometry node group and assign it to the active modifier
    bpy.ops.node.new_geometry_node_group_assign()
    node_tree = modifier.node_group
    out_node = node_tree.nodes["Group Output"]
    in_node = node_tree.nodes["Group Input"]
    node_x_location = 0
    node_location_step_x = 300

    target_info_node, node_x_location = create_node(
        node_tree, "GeometryNodeObjectInfo", node_x_location, node_location_step_x
    )
    target_info_node.inputs[0].default_value = target_object
    target_info_node.transform_space = "RELATIVE"

    boolean_node, node_x_location = create_node(
        node_tree, "GeometryNodeMeshBoolean", node_x_location, node_location_step_x
    )

    boolean_node.operation = "INTERSECT"
    out_node.location.x = node_x_location

    link_nodes_by_socket(
        node_tree,
        from_node=in_node,
        from_socket="Geometry",
        to_node=boolean_node,
        to_socket="Mesh 2",
    )

    link_nodes_by_socket(
        node_tree,
        from_node=target_info_node,
        from_socket="Geometry",
        to_node=boolean_node,
        to_socket="Mesh 2",
    )

    link_nodes_by_socket(
        node_tree,
        from_node=boolean_node,
        from_socket="Mesh",
        to_node=out_node,
        to_socket="Geometry",
    )


def is_intersecting(source_object):
    """Check if an object is intersecting with another object."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = source_object.evaluated_get(depsgraph)
    mesh_eval = obj_eval.data
    return len(mesh_eval.vertices.values()) > 0


def inside_mesh(x, y, z, mesh):
    """Check if a point is inside a mesh."""
    p = Vector((x, y, z))
    max_dist = 1.0e20
    hit, point, normal, face = mesh.closest_point_on_mesh(p, distance=max_dist)
    p2 = point - p
    v = p2.dot(normal)
    return not (v < 0.0)


def generate_random_light_dataset(start_index, n_images, random_sphere=False, use_gpu=True, show_progress=True):
    """
    Generate a dataset of random light positions in the scene.
    
    Args:
        start_index (int): The starting index for the dataset.
        n_images (int): The number of images to generate.
        random_sphere (bool): Whether to randomly position a sphere in the scene.
        use_gpu (bool): Whether to use GPU rendering if available.
        show_progress (bool): Whether to show progress information.
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
    large_box = bpy.data.objects["large_box"]
    small_box = bpy.data.objects["small_box"]
    sphere = bpy.data.objects["sphere"]
    
    # Set up sphere if random_sphere is True
    if random_sphere:
        sphere.hide_render = False
        sphere.hide_viewport = False
        # Create geo nodegroup to test for intersections
        create_intersection_nodegroup(sphere, large_box)
    
    # Get render nodes
    render_png_node = bpy.data.scenes["Scene"].node_tree.nodes["render_png"]
    render_diffdir_png_node = bpy.data.scenes["Scene"].node_tree.nodes[
        "render_diffdir_png"
    ]
    render_diffindir_png_node = bpy.data.scenes["Scene"].node_tree.nodes[
        "render_diffindir_png"
    ]
    render_diffcol_png_node = bpy.data.scenes["Scene"].node_tree.nodes[
        "render_diffcol_png"
    ]
    render_glossdir_png_node = bpy.data.scenes["Scene"].node_tree.nodes[
        "render_glossdir_png"
    ]
    render_glossindir_png_node = bpy.data.scenes["Scene"].node_tree.nodes[
        "render_glossindir_png"
    ]
    render_glosscol_png_node = bpy.data.scenes["Scene"].node_tree.nodes[
        "render_glosscol_png"
    ]
    
    # Define bounds for random positioning
    x_range = [-0.23, 0.23]
    y_range = [-0.07, 0.38]
    z_range = [-0.23, 0.23]

    count = start_index
    modifier = sphere.modifiers.get("Geometry Nodes")
    
    if show_progress:
        print(f"Generating {n_images} random light images...")
    
    while count < n_images:
        if random_sphere:
            modifier.show_render = True
            modifier.show_viewport = True
            is_sphere_intersecting = True
            while is_sphere_intersecting:
                # Randomize sphere location, until it does not intersect boxes
                x = np.random.uniform(x_range[0], x_range[1])
                y = np.random.uniform(y_range[0], y_range[1])
                z = np.random.uniform(z_range[0], z_range[1])
                sphere.location = (x, y, z)
                set_intersection_target(sphere, large_box)
                is_sphere_intersecting = False
                if is_intersecting(sphere):
                    is_sphere_intersecting = True
                set_intersection_target(sphere, small_box)
                if is_intersecting(sphere):
                    is_sphere_intersecting = True
            modifier.show_render = False
            modifier.show_viewport = False
        
        # Position the light
        is_light_inside_mesh = True
        while is_light_inside_mesh:
            # Randomize point light location in reasonable bounds
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            z = np.random.uniform(z_range[0], z_range[1])
            if (
                inside_mesh(x, y, z, large_box)
                or inside_mesh(x, y, z, small_box)
                or inside_mesh(x, y, z, sphere)
            ):
                is_light_inside_mesh = True
            else:
                is_light_inside_mesh = False
        
        bpy.data.objects["Point"].location = (x, y, z)
        
        # Set up file paths
        render_png_node.file_slots[0].path = f"{count:05d}_render_"
        render_diffdir_png_node.file_slots[0].path = f"{count:05d}_diffdir_"
        render_diffindir_png_node.file_slots[0].path = f"{count:05d}_diffindir_"
        render_diffcol_png_node.file_slots[0].path = f"{count:05d}_diffcol_"
        render_glossdir_png_node.file_slots[0].path = f"{count:05d}_glossdir_"
        render_glossindir_png_node.file_slots[0].path = f"{count:05d}_glossindir_"
        render_glosscol_png_node.file_slots[0].path = f"{count:05d}_glosscol_"
        
        # Render image, dirdiffuse, albedo, and indirdiff as pngs
        bpy.ops.render.render()
        
        # Show progress
        if show_progress and (count - start_index + 1) % 10 == 0 or count == n_images - 1:
            print(f"Generated {count - start_index + 1}/{n_images} images ({(count - start_index + 1) / n_images * 100:.1f}%)")
        
        count += 1
    
    # Reset scene to default
    set_default_scene()
    
    if show_progress:
        print("Random light dataset generation complete.") 