import bpy
from pathlib import Path
import numpy as np
from PIL import Image
from pathlib import Path
from mathutils import Vector
import OpenEXR
import Imath


ROOT_PATH = Path(bpy.path.abspath("//"))
BF_PATH = "images"
BF_RES_PATH = Path(ROOT_PATH, "images", "brute_force_result")


def get_active_object():
    return bpy.context.view_layer.objects.active


def clean_modifiers(object):
    object.modifiers.clear()


def clean_nodegroups():
    for i in range(len(bpy.data.node_groups)):
        bpy.data.node_groups.remove(bpy.data.node_groups[0])


def set_default_scene():
    bpy.context.scene.render.resolution_x = 1024
    bpy.context.scene.render.resolution_y = 1024
    bpy.data.objects["Point"].hide_render = True
    bpy.data.objects["Point"].hide_viewport = True
    bpy.data.objects["Area1"].hide_render = False
    bpy.data.objects["Area1"].hide_viewport = False

    render_png_node = bpy.data.scenes["Scene"].node_tree.nodes["render_png"]
    render_png_node.base_path = str(Path(ROOT_PATH, "images"))
    render_png_node.file_slots[0].path = "render"

    render_diffdir_png_node = bpy.data.scenes["Scene"].node_tree.nodes[
        "render_diffdir_png"
    ]
    render_diffdir_png_node.base_path = str(Path(ROOT_PATH, "images"))
    render_diffdir_png_node.file_slots[0].path = "diffdir"

    render_diffindir_png_node = bpy.data.scenes["Scene"].node_tree.nodes[
        "render_diffindir_png"
    ]
    render_diffindir_png_node.base_path = str(Path(ROOT_PATH, "images"))
    render_diffindir_png_node.file_slots[0].path = "diffindir"

    render_diffcol_png_node = bpy.data.scenes["Scene"].node_tree.nodes[
        "render_diffcol_png"
    ]
    render_diffcol_png_node.base_path = str(Path(ROOT_PATH, "images"))
    render_diffcol_png_node.file_slots[0].path = "diffcol"
    sphere = bpy.data.objects["sphere"]
    sphere.hide_render = False
    sphere.hide_viewport = False
    clean_modifiers(sphere)
    clean_nodegroups()


def get_scene_resolution(scene):
    resolution_scale = scene.render.resolution_percentage / 100.0
    resolution_x = scene.render.resolution_x * resolution_scale  # [pixels]
    resolution_y = scene.render.resolution_y * resolution_scale  # [pixels]
    return int(resolution_x), int(resolution_y)


def remove_texures():
    # remove current textures
    exceptions = ["marble.jpg"]
    string_exceptions = ["Wall"]
    for image in bpy.data.images:
        if image.name not in exceptions:
            if ~np.any(np.array([x in image.name for x in string_exceptions])):
                print("removing texture: {}".format(image))
                bpy.data.images.remove(image)


def load_texture(texture_path, texture_key, proj_h, proj_w, overwrite=True):
    for image in bpy.data.images:
        if overwrite and image.name == texture_key:
            bpy.data.images.remove(image)
    pilImage = Image.open(str(texture_path)).convert("RGB")
    image = np.asarray(pilImage)
    if image.shape[0] != proj_h or image.shape[1] != proj_w:
        pilImage = pilImage.resize((proj_w, proj_h))
        image = np.asarray(pilImage)
    float_texture = (image / 255).astype(np.float32)
    # flipped_texture = np.flip(float_texture, axis=0)
    padded_texture = np.concatenate(
        (float_texture, np.ones_like(float_texture)[:, :, 0:1]), axis=-1
    )
    bpy_image = bpy.data.images.new(
        texture_key, width=proj_w, height=proj_h, alpha=False
    )
    bpy_image.pixels.foreach_set(padded_texture.ravel())
    # bpy_image.pack()


def save_texture(texture_key, proj_width, proj_height, dst):
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
    projector_name = "Projector"
    bpy.data.images[texture_name].colorspace_settings.name = "Linear"
    bpy.data.images[texture_name].source = "FILE"
    bpy.data.images[texture_name].filepath = str(
        Path(ROOT_PATH, BF_PATH, texture_name + ".png")
    )
    # bpy.data.lights[projector_name].node_tree.nodes["Image Texture"].interpolation = 'Closest'
    bpy.data.lights[projector_name].node_tree.nodes["Image Texture"].image = (
        bpy.data.images[texture_name]
    )


def hide_object_and_children(obj, hide=True):
    # hide the children
    obj.hide_viewport = hide
    obj.hide_render = hide
    for child in obj.children:
        child.hide_viewport = hide
        child.hide_render = hide


def create_brute_force_images(resx, resy, overwrite=False):
    for y in range(resy):
        for x in range(resx):
            filename = Path(ROOT_PATH, BF_PATH, "BF_{:05d}.png".format(y * resx + x))
            if overwrite or not filename.exists():
                img_np = np.zeros((resy, resx, 3), dtype=np.uint8)
                img_np[y, x] = 255
                image = Image.fromarray(img_np)
                image.save(str(filename))


def insideMesh(x, y, z, mesh):
    p = Vector((x, y, z))
    max_dist = 1.0e20
    hit, point, normal, face = mesh.closest_point_on_mesh(p, distance=max_dist)
    p2 = point - p
    v = p2.dot(normal)
    return not (v < 0.0)


def transport_matrix(proj_resx=64, proj_resy=64, overwrite=False, batch_size=100, show_progress=True):
    """
    Generate a transport matrix by rendering the scene with each projector pixel lit individually.
    
    Args:
        proj_resx (int): Resolution of the projector in the x direction.
        proj_resy (int): Resolution of the projector in the y direction.
        overwrite (bool): Whether to overwrite existing brute force images.
        batch_size (int): Number of images to render in a batch before saving.
        show_progress (bool): Whether to show progress information.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Get the scene resolution
        resx, resy = get_scene_resolution(bpy.context.scene)
        
        # Create brute force images if they don't exist or if overwrite is True
        if show_progress:
            print(f"Creating brute force images ({proj_resx}x{proj_resy})...")
        create_brute_force_images(proj_resx, proj_resy, overwrite=overwrite)
        
        # Calculate the total number of images to render
        total_images = proj_resx * proj_resy
        
        # Create the output directory if it doesn't exist
        BF_RES_PATH.mkdir(parents=True, exist_ok=True)
        
        # Render each image
        if show_progress:
            print(f"Rendering {total_images} images...")
        
        for i in range(total_images):
            # Calculate x and y coordinates
            y = i // proj_resx
            x = i % proj_resx
            
            # Construct the texture path and key
            texture_path = Path(ROOT_PATH, BF_PATH, f"BF_{i:05d}.png")
            texture_key = f"BF_{i:05d}"
            
            # Load and swap the texture
            load_texture(texture_path, texture_key, proj_resx, proj_resy)
            swap_projector_texture(texture_key)
            
            # Render the scene
            bpy.ops.render.render()
            
            # Rename the rendered file
            render_path = Path(BF_RES_PATH, "render0000.exr")
            target_path = Path(BF_RES_PATH, f"BF_{i:05d}.exr")
            
            # Check if the render was successful
            if not render_path.exists():
                print(f"Warning: Render failed for image {i}. Skipping.")
                continue
                
            # Rename the file
            render_path.rename(target_path)
            
            # Show progress
            if show_progress and (i + 1) % 10 == 0 or i == total_images - 1:
                print(f"Rendered {i + 1}/{total_images} images ({(i + 1) / total_images * 100:.1f}%)")
        
        if show_progress:
            print("Transport matrix generation complete.")
        
        return True
    
    except Exception as e:
        print(f"Error in transport_matrix: {e}")
        return False


def create_node(node_tree, type_name, node_x_location, node_location_step_x=0):
    """Creates a node of a given type, and sets/updates the location of the node on the X axis.
    Returning the node object and the next location on the X axis for the next node.
    """
    node_obj = node_tree.nodes.new(type=type_name)
    node_obj.location.x = node_x_location
    node_x_location += node_location_step_x

    return node_obj, node_x_location


def link_nodes_by_socket(node_tree, from_node, from_socket, to_node, to_socket):
    node_tree.links.new(from_node.outputs[from_socket], to_node.inputs[to_socket])


def set_intersection_target(source_object, target_object):
    # this function assumes create_intersection_nodegroup was called on source_object
    modifier = source_object.modifiers.get("Geometry Nodes")
    target_info_node = modifier.node_group.nodes["Object Info"]
    target_info_node.inputs[0].default_value = target_object


def create_intersection_nodegroup(source_object, target_object):
    # clear modifiers and all node groups
    clean_modifiers(source_object)
    clean_nodegroups()
    # create geo node modifier
    modifier = source_object.modifiers.new(type="NODES", name="Geometry Nodes")
    # modifier =  source_object.modifiers.get("Geometry Nodes")
    
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
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = source_object.evaluated_get(depsgraph)
    mesh_eval = obj_eval.data
    return len(mesh_eval.vertices.values()) > 0


def random_point_light(start_index, n_images, random_sphere=False):
    # set up the scene
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.data.objects["Area1"].hide_render = True
    bpy.data.objects["Area1"].hide_viewport = True
    # enable point light
    bpy.data.objects["Point"].hide_render = False
    bpy.data.objects["Point"].hide_viewport = False
    large_box = bpy.data.objects["large_box"]
    small_box = bpy.data.objects["small_box"]
    sphere = bpy.data.objects["sphere"]
    if random_sphere:
        sphere.hide_render = False
        sphere.hide_viewport = False
        # create geo nodegroup to test for intersections
        create_intersection_nodegroup(sphere, large_box)
        # set_intersection_target(sphere, small_box)
        # print(is_intersecting(sphere))
        # modifier = sphere.modifiers.get("Geometry Nodes")
        # modifier.show_render = False
        # modifier.show_viewport = False
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
    x_range = [-0.23, 0.23]
    y_range = [-0.07, 0.38]
    z_range = [-0.23, 0.23]

    count = start_index
    modifier = sphere.modifiers.get("Geometry Nodes")
    while count < n_images:
        if random_sphere:
            modifier.show_render = True
            modifier.show_viewport = True
            is_sphere_intersecting = True
            while is_sphere_intersecting:
                # randomize sphere location, until it does not intersect boxes
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
        is_light_inside_mesh = True
        while is_light_inside_mesh:
            # randomize point light location in reasonble bounds
            x = np.random.uniform(x_range[0], x_range[1])
            y = np.random.uniform(y_range[0], y_range[1])
            z = np.random.uniform(z_range[0], z_range[1])
            if (
                insideMesh(x, y, z, large_box)
                or insideMesh(x, y, z, small_box)
                or insideMesh(x, y, z, sphere)
            ):
                is_light_inside_mesh = True
            else:
                is_light_inside_mesh = False
        bpy.data.objects["Point"].location = (x, y, z)
        render_png_node.file_slots[0].path = "{:05d}_render_".format(count)
        render_diffdir_png_node.file_slots[0].path = "{:05d}_diffdir_".format(count)
        render_diffindir_png_node.file_slots[0].path = "{:05d}_diffindir_".format(count)
        render_diffcol_png_node.file_slots[0].path = "{:05d}_diffcol_".format(count)
        render_glossdir_png_node.file_slots[0].path = "{:05d}_glossdir_".format(count)
        render_glossindir_png_node.file_slots[0].path = "{:05d}_glossindir_".format(
            count
        )
        render_glosscol_png_node.file_slots[0].path = "{:05d}_glosscol_".format(count)
        # render image, dirdiffuse, albedo, and indirdiff as pngs
        bpy.ops.render.render()
        count += 1
    set_default_scene()


def calculate_transport_matrix(proj_resx=64, proj_resy=64, batch_size=100, show_progress=True, use_multiprocessing=False):
    """
    Calculate the transport matrix from the rendered EXR images.
    
    Args:
        proj_resx (int): Resolution of the projector in the x direction.
        proj_resy (int): Resolution of the projector in the y direction.
        batch_size (int): Number of images to process in a batch.
        show_progress (bool): Whether to show progress information.
        use_multiprocessing (bool): Whether to use multiprocessing for faster computation.
    
    Returns:
        numpy.ndarray: The transport matrix where each row represents the response
                      of the scene to a single projector pixel.
    """
    try:
        # Get the number of projector pixels
        num_projector_pixels = proj_resx * proj_resy
        
        # Get the scene resolution
        resx, resy = get_scene_resolution(bpy.context.scene)
        num_scene_pixels = resx * resy
        
        # Initialize the transport matrix
        # Each row represents the response of the scene to a single projector pixel
        transport_matrix = np.zeros((num_projector_pixels, num_scene_pixels, 3), dtype=np.float32)
        
        # Check if the output directory exists
        if not BF_RES_PATH.exists():
            print(f"Error: Output directory {BF_RES_PATH} does not exist.")
            print("Please run transport_matrix() first to generate the EXR files.")
            return None
        
        # Count the number of existing EXR files
        existing_files = list(BF_RES_PATH.glob("BF_*.exr"))
        if len(existing_files) < num_projector_pixels:
            print(f"Warning: Only {len(existing_files)}/{num_projector_pixels} EXR files found.")
            print("Some files may be missing. The transport matrix will be incomplete.")
        
        # Process the EXR files
        if show_progress:
            print(f"Processing {len(existing_files)} EXR files...")
        
        # Function to process a single EXR file
        def process_exr_file(file_index):
            exr_path = Path(BF_RES_PATH, f"BF_{file_index:05d}.exr")
            if not exr_path.exists():
                if show_progress:
                    print(f"Warning: EXR file {exr_path} does not exist. Skipping.")
                return None
                
            try:
                # Open the EXR file
                exr_file = OpenEXR.InputFile(str(exr_path))
                
                # Get the data window
                dw = exr_file.header()['dataWindow']
                size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
                
                # Read the RGB channels
                FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
                (R,G,B) = [np.frombuffer(exr_file.channel(Chan, FLOAT), dtype=np.float32) for Chan in ("R", "G", "B")]
                
                # Reshape the channels to match the image dimensions
                R = R.reshape(size[1], size[0])
                G = G.reshape(size[1], size[0])
                B = B.reshape(size[1], size[0])
                
                # Combine the channels
                img = np.stack([R, G, B], axis=-1)
                
                # Flatten the image to get a 1D array of RGB values
                img_flat = img.reshape(-1, 3)
                
                # If the flattened image has fewer pixels than expected, pad with zeros
                if img_flat.shape[0] < num_scene_pixels:
                    padding = np.zeros((num_scene_pixels - img_flat.shape[0], 3), dtype=np.float32)
                    img_flat = np.vstack([img_flat, padding])
                
                # Close the EXR file
                exr_file.close()
                
                return img_flat[:num_scene_pixels, :]
                
            except Exception as e:
                if show_progress:
                    print(f"Error processing {exr_path}: {e}")
                return None
        
        # Process the EXR files
        if use_multiprocessing:
            # Use multiprocessing for faster computation
            import multiprocessing as mp
            num_processes = mp.cpu_count()
            
            if show_progress:
                print(f"Using {num_processes} processes for parallel computation...")
            
            with mp.Pool(processes=num_processes) as pool:
                results = pool.map(process_exr_file, range(num_projector_pixels))
                
                # Update the transport matrix with the results
                for i, result in enumerate(results):
                    if result is not None:
                        transport_matrix[i, :, :] = result
                    
                    # Show progress
                    if show_progress and (i + 1) % batch_size == 0 or i == num_projector_pixels - 1:
                        print(f"Processed {i + 1}/{num_projector_pixels} EXR files ({(i + 1) / num_projector_pixels * 100:.1f}%)")
        else:
            # Process the EXR files sequentially
            for i in range(num_projector_pixels):
                result = process_exr_file(i)
                if result is not None:
                    transport_matrix[i, :, :] = result
                
                # Show progress
                if show_progress and (i + 1) % batch_size == 0 or i == num_projector_pixels - 1:
                    print(f"Processed {i + 1}/{num_projector_pixels} EXR files ({(i + 1) / num_projector_pixels * 100:.1f}%)")
        
        if show_progress:
            print("Transport matrix calculation complete.")
        
        return transport_matrix
    
    except Exception as e:
        print(f"Error in calculate_transport_matrix: {e}")
        return None


def save_transport_matrix(transport_matrix, output_path=None):
    """
    Save the transport matrix to a file.
    
    Args:
        transport_matrix (numpy.ndarray): The transport matrix to save.
        output_path (str, optional): The path to save the transport matrix to.
                                    If None, a default path will be used.
    """
    if output_path is None:
        output_path = Path(ROOT_PATH, "transport_matrix.npy")
    
    # Save the transport matrix
    np.save(output_path, transport_matrix)
    print(f"Transport matrix saved to {output_path}")


def load_transport_matrix(input_path=None):
    """
    Load a transport matrix from a file.
    
    Args:
        input_path (str, optional): The path to load the transport matrix from.
                                   If None, a default path will be used.
    
    Returns:
        numpy.ndarray: The loaded transport matrix.
    """
    if input_path is None:
        input_path = Path(ROOT_PATH, "transport_matrix.npy")
    
    # Load the transport matrix
    transport_matrix = np.load(input_path)
    print(f"Transport matrix loaded from {input_path}")
    
    return transport_matrix


set_default_scene()
random_point_light(0, 1, random_sphere=True)
