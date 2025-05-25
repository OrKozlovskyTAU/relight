import bpy
from pathlib import Path
import numpy as np
from PIL import Image
import OpenEXR
import Imath
import multiprocessing as mp

from relight.utils.blender_utils import (
    get_scene_resolution,
    load_texture,
    swap_projector_texture,
    setup_gpu_rendering
)


def create_brute_force_images(resx, resy, output_dir, overwrite=False):
    """
    Create brute force images for transport matrix calculation.
    
    Args:
        resx (int): Resolution of the projector in the x direction.
        resy (int): Resolution of the projector in the y direction.
        output_dir (Path): Directory to save the images.
        overwrite (bool): Whether to overwrite existing images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for y in range(resy):
        for x in range(resx):
            filename = output_dir / f"BF_{y * resx + x:05d}.png"
            if overwrite or not filename.exists():
                img_np = np.zeros((resy, resx, 3), dtype=np.uint8)
                img_np[y, x] = 255
                image = Image.fromarray(img_np)
                image.save(str(filename))


def generate_transport_matrix(proj_resx=64, proj_resy=64, overwrite=False, batch_size=100, show_progress=True, use_gpu=True):
    """
    Generate a transport matrix by rendering the scene with each projector pixel lit individually.
    
    Args:
        proj_resx (int): Resolution of the projector in the x direction.
        proj_resy (int): Resolution of the projector in the y direction.
        overwrite (bool): Whether to overwrite existing brute force images.
        batch_size (int): Number of images to render in a batch before saving.
        show_progress (bool): Whether to show progress information.
        use_gpu (bool): Whether to use GPU rendering if available.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Set up GPU rendering if requested
        if use_gpu:
            setup_gpu_rendering()
        
        # Get the scene resolution
        resx, resy = get_scene_resolution(bpy.context.scene)
        
        # Set up paths
        root_path = Path(bpy.path.abspath("//"))
        bf_path = root_path / "images"
        bf_res_path = bf_path / "brute_force_result"
        
        # Create brute force images if they don't exist or if overwrite is True
        if show_progress:
            print(f"Creating brute force images ({proj_resx}x{proj_resy})...")
        create_brute_force_images(proj_resx, proj_resy, bf_path, overwrite=overwrite)
        
        # Calculate the total number of images to render
        total_images = proj_resx * proj_resy
        
        # Create the output directory if it doesn't exist
        bf_res_path.mkdir(parents=True, exist_ok=True)
        
        # Render each image
        if show_progress:
            print(f"Rendering {total_images} images...")
        
        for i in range(total_images):
            # Calculate x and y coordinates
            y = i // proj_resx
            x = i % proj_resx
            
            # Construct the texture path and key
            texture_path = bf_path / f"BF_{i:05d}.png"
            texture_key = f"BF_{i:05d}"
            
            # Load and swap the texture
            load_texture(texture_path, texture_key, proj_resx, proj_resy)
            swap_projector_texture(texture_key)
            
            # Render the scene
            bpy.ops.render.render()
            
            # Rename the rendered file
            render_path = bf_res_path / "render0000.exr"
            target_path = bf_res_path / f"BF_{i:05d}.exr"
            
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
        print(f"Error in generate_transport_matrix: {e}")
        return False


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
        # Set up paths
        root_path = Path(bpy.path.abspath("//"))
        bf_res_path = root_path / "images" / "brute_force_result"
        
        # Get the number of projector pixels
        num_projector_pixels = proj_resx * proj_resy
        
        # Get the scene resolution
        resx, resy = get_scene_resolution(bpy.context.scene)
        num_scene_pixels = resx * resy
        
        # Initialize the transport matrix
        # Each row represents the response of the scene to a single projector pixel
        transport_matrix = np.zeros((num_projector_pixels, num_scene_pixels, 3), dtype=np.float32)
        
        # Check if the output directory exists
        if not bf_res_path.exists():
            print(f"Error: Output directory {bf_res_path} does not exist.")
            print("Please run generate_transport_matrix() first to generate the EXR files.")
            return None
        
        # Count the number of existing EXR files
        existing_files = list(bf_res_path.glob("BF_*.exr"))
        if len(existing_files) < num_projector_pixels:
            print(f"Warning: Only {len(existing_files)}/{num_projector_pixels} EXR files found.")
            print("Some files may be missing. The transport matrix will be incomplete.")
        
        # Process the EXR files
        if show_progress:
            print(f"Processing {len(existing_files)} EXR files...")
        
        # Function to process a single EXR file
        def process_exr_file(file_index):
            exr_path = bf_res_path / f"BF_{file_index:05d}.exr"
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
        root_path = Path(bpy.path.abspath("//"))
        output_path = root_path / "transport_matrix.npy"
    
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
        root_path = Path(bpy.path.abspath("//"))
        input_path = root_path / "transport_matrix.npy"
    
    # Load the transport matrix
    transport_matrix = np.load(input_path)
    print(f"Transport matrix loaded from {input_path}")
    
    return transport_matrix 