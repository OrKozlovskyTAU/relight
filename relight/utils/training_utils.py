"""
Common training utilities for Relight models.

This module contains utility functions used across different training scripts.
"""

import logging
import os
import shutil
from pathlib import Path

import accelerate
import diffusers
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from color_transfer import color_transfer
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from packaging import version

logger = get_logger(__name__)

def unwrap_model(accelerator, model):
    """Unwrap a model from accelerator and compiled wrappers."""
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

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

def setup_optimizer(args, controlnet):
    """Set up the optimizer for training."""
        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = controlnet.parameters()
    # use adafactor optimizer to save gpu memory
    if args.use_adafactor:
        from transformers import Adafactor

        optimizer = Adafactor(
            params_to_optimize,
            lr=args.learning_rate,
            scale_parameter=False,
            relative_step=False,
            # warmup_init=True,
            weight_decay=args.adam_weight_decay,
        )
    else:
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    
    return optimizer

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
    """Create output directory."""
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

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
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    
    return True

def create_model_hooks(accelerator, args, models, model_type="sd3"):
    """
    Create model hooks for saving and loading models.
    
    Args:
        accelerator: The accelerator instance
        args: Training arguments
        models: List of models to save/load
        model_type: Either "sd3" or "flux" to determine which model type to use
        
    Returns:
        Tuple of (save_model_hook, load_model_hook) functions
    """
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet" if model_type == "sd3" else "flux_controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                if model_type == "sd3":
                    from diffusers import SD3ControlNetModel
                    load_model = SD3ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                else:
                    from diffusers.models.controlnets.controlnet_flux import (
                        FluxControlNetModel,
                    )
                    load_model = FluxControlNetModel.from_pretrained(input_dir, subfolder="flux_controlnet")
                
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
        
        return save_model_hook, load_model_hook
    
    return None, None

def setup_weight_dtype(args, accelerator):
    """Set up the weight dtype for training."""
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    return weight_dtype
   
def color_match_lab(pred, inp):
    """
    Color-match the prediction to the input image in LAB color space using Reinhard's method.
    Args:
        pred: np.ndarray, shape (H, W, 3), RGB, range [0, 1] or [0, 255]
        inp: np.ndarray, shape (H, W, 3), RGB, range [0, 1] or [0, 255]
    Returns:
        np.ndarray, shape (H, W, 3), RGB, same dtype/range as input
    """
    # Ensure float and [0, 1] range for color_transfer
    pred = pred.astype(np.float32)
    inp = inp.astype(np.float32)
    if pred.max() > 1.1 or inp.max() > 1.1:
        pred = pred / 255.0
        inp = inp / 255.0

    matched = color_transfer(source=inp, target=pred)
    # Convert back to [0, 255] if needed
    if inp.max() > 1.1:
        matched = (matched * 255).astype(np.uint8)
    return matched
