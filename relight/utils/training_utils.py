"""
Common training utilities for Relight models.

This module contains utility functions used across different training scripts.
"""

import os
import shutil
import logging
import torch
import numpy as np
from pathlib import Path
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.utils import make_image_grid, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from huggingface_hub import create_repo
from accelerate import DistributedDataParallelKwargs, ProjectConfiguration
import transformers
import diffusers

logger = get_logger(__name__)

def unwrap_model(accelerator, model):
    """Unwrap a model from accelerator and compiled wrappers."""
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    """Save a model card with example images."""
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            make_image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "flux",
        "flux-diffusers",
        "text-to-image",
        "diffusers",
        "controlnet",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))

def get_sigmas(timesteps, noise_scheduler_copy, n_dim=4, dtype=torch.float32, device=None):
    """Get sigma values for timesteps."""
    sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

def setup_accelerator(args):
    """Set up the accelerator for training."""
    logging_dir = Path(args.output_dir, args.logging_dir)
    
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, 
        logging_dir=logging_dir
    )
    
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    
    # Disable AMP for MPS
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
        
    return accelerator

def setup_logging(accelerator, args):
    """Set up logging for training."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
        
    # Set the training seed
    if args.seed is not None:
        set_seed(args.seed)
        
    return logger

def create_output_dir(args, accelerator):
    """Create output directory and handle repository creation."""
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, 
                exist_ok=True, 
                token=args.hub_token
            ).repo_id
            return repo_id
    return None

def save_checkpoint(accelerator, args, global_step, checkpoints_total_limit=None):
    """Save a checkpoint of the training state."""
    if args.checkpoints_total_limit is not None:
        checkpoints = os.listdir(args.output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)
    logger.info(f"Saved state to {save_path}")

def validate_training_args(args):
    """Validate training arguments."""
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    
    return True

def create_model_hooks(accelerator, args, models, optimizer, lr_scheduler):
    """Create model hooks for saving and loading checkpoints."""
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                unwrapped_model = unwrap_model(model)
                unwrapped_model.save_pretrained(
                    output_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                    state_dict=accelerator.get_state_dict(model),
                )
                if hasattr(unwrapped_model, "save_config"):
                    unwrapped_model.save_config(output_dir)
    
    def load_model_hook(models, input_dir):
        for i, model in enumerate(models):
            unwrapped_model = unwrap_model(model)
            unwrapped_model.load_pretrained(
                input_dir,
                is_main_process=accelerator.is_main_process,
                load_function=accelerator.load,
            )
    
    # Register hooks
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    return save_model_hook, load_model_hook

def compute_text_embeddings(batch, text_encoders, tokenizers, proportion_empty_prompts=0, is_train=True, max_sequence_length=77, return_dict=False):
    """Compute text embeddings for a batch.
    
    Args:
        batch: The batch containing prompts
        text_encoders: List of text encoders
        tokenizers: List of tokenizers
        proportion_empty_prompts: Proportion of empty prompts for classifier-free guidance
        is_train: Whether in training mode
        max_sequence_length: Maximum sequence length for tokenization
        return_dict: Whether to return a dictionary with embeddings or tuple
        
    Returns:
        If return_dict is True, returns a dictionary with prompt_embeds and pooled_prompt_embeds
        Otherwise, returns a tuple of (prompt_embeds, pooled_embeds, uncond_embeds)
    """
    # Get the text embeddings for conditioning
    prompt_key = "prompts" if "prompts" in batch else "prompt"
    prompt = batch[prompt_key]
    
    prompt_embeds, pooled_embeds = encode_prompt(
        text_encoders,
        tokenizers,
        prompt,
        max_sequence_length=max_sequence_length,
        device=text_encoders[0].device,
        num_images_per_prompt=1,
    )
    
    # Get the unconditional embeddings for classifier-free guidance
    if proportion_empty_prompts > 0:
        uncond_tokens = [""] * (batch[prompt_key].size(0) if isinstance(batch[prompt_key], torch.Tensor) else len(batch[prompt_key]))
        uncond_embeds, _ = encode_prompt(
            text_encoders,
            tokenizers,
            uncond_tokens,
            max_sequence_length=max_sequence_length,
            device=text_encoders[0].device,
            num_images_per_prompt=1,
        )
    else:
        uncond_embeds = None
    
    if return_dict:
        return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_embeds}
    else:
        return prompt_embeds, pooled_embeds, uncond_embeds

def encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length, device=None, num_images_per_prompt=1):
    """Encode prompt using text encoders."""
    prompt_embeds = []
    pooled_embeds = []
    
    for text_encoder, tokenizer in zip(text_encoders, tokenizers):
        if isinstance(text_encoder, transformers.models.clip.modeling_clip.CLIPTextModel):
            prompt_embeds_one, pooled_embeds_one = _encode_prompt_with_clip(
                text_encoder, tokenizer, prompt, device, num_images_per_prompt
            )
            prompt_embeds.append(prompt_embeds_one)
            pooled_embeds.append(pooled_embeds_one)
        elif isinstance(text_encoder, transformers.models.t5.modeling_t5.T5EncoderModel):
            prompt_embeds_one = _encode_prompt_with_t5(
                text_encoder, tokenizer, prompt, max_sequence_length, device, num_images_per_prompt
            )
            prompt_embeds.append(prompt_embeds_one)
            pooled_embeds.append(None)
    
    return prompt_embeds, pooled_embeds

def _encode_prompt_with_clip(text_encoder, tokenizer, prompt, device=None, num_images_per_prompt=1):
    """Encode prompt using CLIP text encoder."""
    batch_size = len(prompt)
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    
    with torch.no_grad():
        text_outputs = text_encoder(text_input_ids)
        pooled_embeds = text_outputs[0]
        prompt_embeds = text_outputs[1]
    
    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    pooled_embeds = pooled_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    
    return prompt_embeds, pooled_embeds

def _encode_prompt_with_t5(text_encoder, tokenizer, prompt, max_sequence_length, device=None, num_images_per_prompt=1):
    """Encode prompt using T5 text encoder."""
    batch_size = len(prompt)
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    
    with torch.no_grad():
        text_outputs = text_encoder(text_input_ids)
        prompt_embeds = text_outputs[0]
    
    prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
    
    return prompt_embeds 