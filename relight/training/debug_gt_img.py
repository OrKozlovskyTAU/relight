import os
from pathlib import Path
import torch
from diffusers import AutoencoderKL
from relight.training.dataset import RelightDataset
import torchvision.transforms as T
from PIL import Image

# ---- CONFIG ----
PRETRAINED_MODEL_PATH = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DATA_DIR = "data/train"  # Using training data directory
OUTPUT_DIR = "debug_gt_img_output"
IMAGE_SIZE = 512  # Same as in training script
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_IMAGES = 5

# Set HuggingFace cache directory as in training script
os.environ['HF_HOME'] = '/home/dcor/orkozlovsky/.cache/huggingface'

# ----------------

def main():
    print(f"Using device: {DEVICE}")
    print(f"Loading VAE from: {PRETRAINED_MODEL_PATH}")
    print(f"Loading dataset from: {DATA_DIR}")
    print(f"Saving outputs to: {OUTPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load VAE
    vae = AutoencoderKL.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="vae")
    vae = vae.to(DEVICE)
    vae.eval()

    # Load dataset
    dataset = RelightDataset(
        data_dir=DATA_DIR,
        image_size=IMAGE_SIZE,
        normalize_images=True
    )

    to_pil = T.ToPILImage()

    for idx in range(NUM_IMAGES):
        print(f"\nProcessing image {idx+1}/{NUM_IMAGES}")
        sample = dataset[idx]
        img_tensor = sample["pixel_values"].unsqueeze(0).to(DEVICE)
        print(f"Input tensor shape: {img_tensor.shape}, range: [{img_tensor.min():.2f}, {img_tensor.max():.2f}]")

        # Encode to latents
        with torch.no_grad():
            latents = vae.encode(img_tensor).latent_dist.sample() * vae.config.scaling_factor
            print(f"Latents shape: {latents.shape}, range: [{latents.min():.2f}, {latents.max():.2f}]")
            # Decode as in training
            gt_img = vae.decode(latents / vae.config.scaling_factor).sample
            print(f"Decoded image shape: {gt_img.shape}, range: [{gt_img.min():.2f}, {gt_img.max():.2f}]")

        # Remove batch dim and clamp
        gt_img = gt_img.squeeze(0).detach().cpu()
        gt_img = (gt_img - gt_img.min()) / 2 # denormalize from [-1,1] to [0,1]
        print(f"Denormalized decoded image shape: {gt_img.shape}, range: [{gt_img.min():.2f}, {gt_img.max():.2f}]")
        gt_img = torch.clamp(gt_img, 0, 1)
        
        # Also save the original (denormalized) image
        orig_img = sample["pixel_values"].detach().cpu()
        orig_img = (orig_img + 1) / 2  # denormalize from [-1,1] to [0,1]
        print(f"Denormalized original image shape: {orig_img.shape}, range: [{orig_img.min():.2f}, {orig_img.max():.2f}]")
        orig_img = torch.clamp(orig_img, 0, 1)
        # Compute difference map (absolute difference)
        diff_map = torch.abs(gt_img - orig_img)
        # Optionally, amplify for visualization
        diff_map_vis = torch.clamp(diff_map * 2, 0, 1)

        # Convert to PIL
        gt_img_pil = to_pil(gt_img)
        orig_img_pil = to_pil(orig_img)
        diff_map_pil = to_pil(diff_map_vis)

        # Concatenate side by side
        width, height = gt_img_pil.width, gt_img_pil.height
        combined = Image.new('RGB', (width * 3, height))
        combined.paste(gt_img_pil, (0, 0))
        combined.paste(orig_img_pil, (width, 0))
        combined.paste(diff_map_pil, (width * 2, 0))

        save_path = os.path.join(OUTPUT_DIR, f"comparison_{idx}.png")
        combined.save(save_path)
        print(f"Saved comparison_{idx}.png (GT | ORIG | DIFF)")

if __name__ == "__main__":
    main() 