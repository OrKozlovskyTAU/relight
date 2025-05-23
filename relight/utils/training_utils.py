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
import accelerate
from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel
import copy
from packaging import version

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
            validation_data_dir = log["validation_data_dir"]
            validation_data_dir.save(os.path.join(repo_folder, "image_control.png"))
            images = [validation_data_dir] + images
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

def load_models(args, logger, model_type="sd3"):
    """
    Load essential models for training.
    
    Args:
        args: Training arguments
        logger: Logger instance
        model_type: Either "sd3" or "flux" to determine which models to load
        
    Returns:
        Tuple of (models, schedulers) where models is a tuple of (vae, transformer, controlnet)
        and schedulers is a tuple of (noise_scheduler, noise_scheduler_copy)
    """
    if model_type == "sd3":
        # Load scheduler and models
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            variant=args.variant,
        )
        transformer = SD3Transformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
        )

        if args.controlnet_model_name_or_path:
            logger.info("Loading existing controlnet weights")
            controlnet = SD3ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
        else:
            logger.info("Initializing controlnet weights from transformer")
            controlnet = SD3ControlNetModel.from_transformer(
                transformer, num_extra_conditioning_channels=args.num_extra_conditioning_channels
            )
    else:  # flux
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            variant=args.variant,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            revision=args.revision,
            variant=args.variant,
        )
        if args.controlnet_model_name_or_path:
            logger.info("Loading existing controlnet weights")
            controlnet = FluxControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
        else:
            logger.info("Initializing controlnet weights from transformer")
            controlnet = FluxControlNetModel.from_transformer(
                transformer,
                attention_head_dim=transformer.config["attention_head_dim"],
                num_attention_heads=transformer.config["num_attention_heads"],
                num_layers=args.num_double_layers,
                num_single_layers=args.num_single_layers,
            )
        logger.info("all models loaded successfully")

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="scheduler",
        )
        noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    return (vae, transformer, controlnet), (noise_scheduler, noise_scheduler_copy)

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
                    from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
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

def setup_training(args, accelerator, transformer, vae, controlnet, weight_dtype, model_type="sd3"):
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    controlnet.train()
    
    if model_type == "flux":
        if args.enable_npu_flash_attention:
            if is_torch_npu_available():
                logger.info("npu flash attention enabled.")
                transformer.enable_npu_flash_attention()
            else:
                raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu devices.")

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warning(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                transformer.enable_xformers_memory_efficient_attention()
                controlnet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        if model_type == "flux":
            transformer.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()
    
    # Check that all trainable models are in full precision
    if unwrap_model(controlnet).dtype != torch.float32:
        low_precision_error_string = (
            " Please make sure to always have all model weights in full float32 precision when starting training - even if"
            " doing mixed precision training, copy of the weights should still be float32."
        )
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    
    # Move vae and transformer to device and cast to weight_dtype
    if args.upcast_vae:
        vae.to(accelerator.device, dtype=torch.float32)
    else:
        vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    