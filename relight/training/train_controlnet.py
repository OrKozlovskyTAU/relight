#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

from __future__ import annotations

import contextlib
import gc
import logging
import math
import os
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from PIL import Image
from tqdm.auto import tqdm

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from relight.cli.train import parse_args
from relight.training.dataset import RelightDataset
from relight.utils.wandb_key import get_wandb_key
from dataclasses import dataclass
from typing import Optional

import torchvision.utils as vutils
import torchvision.transforms as T
import numpy as np
from PIL import Image
                
import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.34.0.dev0")

logger = get_logger(__name__)

@dataclass
class ControlNetTrainConfig:
    # Path to the pretrained base model (e.g., Stable Diffusion or similar).
    # This is used as the starting point for training or fine-tuning.
    # Can be a local directory or a HuggingFace model hub identifier.
    pretrained_model_name_or_path: Optional[str] = None

    # Path to a pretrained ControlNet model to resume or fine-tune from.
    # If None, a new ControlNet will be initialized from the base model's UNet.
    # Useful for transfer learning or continuing interrupted training.
    controlnet_model_name_or_path: Optional[str] = None

    # Model variant identifier, such as 'fp16', 'ema', etc.
    # Used to select a specific variant of the model weights (e.g., for mixed precision).
    variant: Optional[str] = None

    # Model revision identifier, such as a git commit hash or branch name.
    # Allows loading a specific version of the model from a repository.
    revision: Optional[str] = None

    # Directory where all outputs, including checkpoints and logs, will be saved.
    # This directory will be created if it does not exist.
    output_dir: str = "controlnet-model"

    # Random seed for reproducibility of training results.
    # Set to a fixed integer for deterministic behavior, or None for random.
    seed: Optional[int] = None

    # The resolution (height and width) to which all input images will be resized.
    # Must be compatible with the model architecture (typically 512 for SD).
    resolution: int = 512

    # Number of samples per batch per device during training.
    # Increasing this can speed up training but requires more GPU memory.
    train_batch_size: int = 4

    # Number of full passes through the training dataset.
    # If max_train_steps is set, this value may be overridden.
    num_train_epochs: int = 1

    # Maximum number of training steps (batches).
    # If set, training will stop after this many steps, regardless of epochs.
    max_train_steps: Optional[int] = None

    # Number of steps between saving model checkpoints.
    # Frequent checkpointing is useful for long runs or unstable training.
    checkpointing_steps: int = 500

    # Maximum number of checkpoints to keep on disk.
    # Older checkpoints will be deleted to save space if this limit is exceeded.
    checkpoints_total_limit: Optional[int] = None

    # Path or keyword to resume training from a specific checkpoint.
    # Use 'latest' to automatically resume from the most recent checkpoint.
    resume_from_checkpoint: Optional[str] = None

    # Number of steps to accumulate gradients before performing an optimizer step.
    # Useful for simulating larger batch sizes on limited hardware.
    gradient_accumulation_steps: int = 1

    # Enable gradient checkpointing to reduce memory usage at the cost of extra computation.
    # This can allow training larger models or using larger batch sizes.
    gradient_checkpointing: bool = False

    # The initial learning rate for the optimizer.
    # May be further adjusted by the learning rate scheduler.
    learning_rate: float = 5e-6

    # If True, scales the learning rate by batch size, accumulation steps, and number of processes.
    # Useful for distributed or large-batch training.
    scale_lr: bool = False

    # Type of learning rate scheduler to use (e.g., 'constant', 'linear', 'cosine').
    # Determines how the learning rate changes during training.
    lr_scheduler: str = "constant"

    # Number of warmup steps for the learning rate scheduler.
    # The learning rate will increase linearly during these steps.
    lr_warmup_steps: int = 500

    # Number of cycles for cosine or similar schedulers.
    # Only relevant for certain scheduler types.
    lr_num_cycles: int = 1

    # Power factor for polynomial learning rate schedulers.
    # Only relevant for certain scheduler types.
    lr_power: float = 1.0

    # Type of mixed precision to use: 'fp16', 'bf16', or None for full precision.
    # Mixed precision can speed up training and reduce memory usage on supported hardware.
    mixed_precision: Optional[str] = None

    # Enable memory-efficient attention using xformers library.
    # Can significantly reduce memory usage for large models.
    enable_xformers_memory_efficient_attention: bool = False

    # If True, uses set_to_none=True when zeroing gradients for potential memory savings.
    set_grads_to_none: bool = False

    # Directory containing the training data (images and annotations).
    # Required if not using a HuggingFace dataset.
    train_data_dir: Optional[str] = None

    # Directory containing validation data for periodic evaluation.
    # If None, validation is skipped during training.
    validation_data_dir: Optional[str] = None

    # Number of images to generate per validation sample during evaluation.
    # Allows for qualitative assessment of model performance.
    num_validation_images: int = 4

    # Number of training steps between validation runs.
    # Validation will be performed every N steps.
    validation_steps: int = 200

    # Name of the project for experiment tracking (e.g., in wandb or tensorboard).
    tracker_project_name: str = "train-controlnet"

    # Reporting backend for experiment tracking (e.g., 'wandb', 'tensorboard').
    # Determines where logs and metrics are sent.
    report_to: str = "wandb"

    # If True, uses 8-bit Adam optimizer for reduced memory usage.
    # Requires the bitsandbytes library and is useful for large models or limited GPUs.
    use_8bit_adam: bool = False

    # Number of worker processes for data loading.
    # Increasing this can speed up data loading but uses more CPU resources.
    dataloader_num_workers: int = 0

    # Beta1 parameter for Adam optimizer (momentum term).
    adam_beta1: float = 0.9

    # Beta2 parameter for Adam optimizer (second moment term).
    adam_beta2: float = 0.999

    # Weight decay (L2 regularization) for Adam optimizer.
    adam_weight_decay: float = 1e-2

    # Epsilon value for Adam optimizer (for numerical stability).
    adam_epsilon: float = 1e-08

    # Maximum gradient norm for gradient clipping.
    # Helps prevent exploding gradients during training.
    max_grad_norm: float = 1.0

    # Directory for logging outputs (e.g., tensorboard logs).
    # Relative to output_dir.
    logging_dir: str = "logs"

    # If True, enables TF32 on Ampere GPUs for faster training.
    # Only has effect on compatible NVIDIA hardware.
    allow_tf32: bool = False

    # Number of inference steps to use during validation image generation.
    # Controls the maximum number of denoising steps for validation samples.
    validation_num_inference_steps: int = 50

    @staticmethod
    def from_args(args) -> ControlNetTrainConfig:
        # Only keep keys that are fields of ControlNetTrainConfig
        valid_keys = set(ControlNetTrainConfig.__dataclass_fields__.keys())
        filtered_args = {k: v for k, v in vars(args).items() if k in valid_keys}
        return ControlNetTrainConfig(**filtered_args)


def log_validation(
    vae, unet, controlnet, config: ControlNetTrainConfig, accelerator, weight_dtype, step, is_final_validation: bool = False
):
    logger.info("Running validation... ")

    if not is_final_validation:
        logger.debug("Using unwrapped controlnet model")
        controlnet = accelerator.unwrap_model(controlnet)
    else:
        logger.info("Loading controlnet from output directory for final validation")
        controlnet = ControlNetModel.from_pretrained(config.output_dir, torch_dtype=weight_dtype)

    logger.info("Creating StableDiffusionControlNetPipeline")
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        vae=vae,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=config.revision,
        variant=config.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if config.enable_xformers_memory_efficient_attention:
        logger.debug("Enabling xformers memory efficient attention")
        pipeline.enable_xformers_memory_efficient_attention()

    if config.seed is None:
        logger.debug("Using random seed for validation")
        generator = None
    else:
        logger.debug(f"Using fixed seed {config.seed} for validation")
        generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)

    logger.info("Loading validation dataset from %s", config.validation_data_dir)
    validation_dataset = RelightDataset(
        data_dir=config.validation_data_dir,
        image_size=config.resolution,
        normalize_images=False  # Don't normalize for PIL display
    )

    image_logs = []
    inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast("cuda")
    logger.debug("Using inference context: %s", type(inference_ctx).__name__)

    for idx, sample in enumerate(validation_dataset):
        logger.debug("Processing validation sample %d/%d", idx + 1, len(validation_dataset))
        control_image_path = os.path.join(config.validation_data_dir, sample['control_file'])
        target_image_path = os.path.join(config.validation_data_dir, sample['target_file'])
        logger.debug("Loading control image from %s", control_image_path)
        control_image = Image.open(control_image_path).convert("RGB")
        target_image = Image.open(target_image_path).convert("RGB")

        images = []
        steps_range = np.linspace(0, config.validation_num_inference_steps, config.num_validation_images, dtype=int)
        for i, num_steps in enumerate(steps_range):
            logger.debug("Generating validation image %d/%d for sample %d with %d steps", 
                        i + 1, config.num_validation_images, idx + 1, num_steps)
            with inference_ctx:
                image = pipeline(
                    "",  # Null prompt
                    control_image,
                    num_inference_steps=num_steps,
                    generator=generator
                ).images[0]
            images.append(image)

        # Combine target image, control image, and generated images
        combined_images = [target_image, control_image] + images
        image_logs.append(
            {"images": combined_images}
        )

    tracker_key = "test" if is_final_validation else "validation"
    logger.info("Logging validation results to trackers")
    for tracker in accelerator.trackers:
        logger.debug("Logging to tracker: %s", tracker.name)
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                formatted_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images("validation", formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []
            for log in image_logs:
                images = log["images"]
                # Concatenate images horizontally
                concat_image = np.concatenate([np.asarray(img) for img in images], axis=1)
                # Log both individual and concatenated images
                formatted_images.append(
                    wandb.Image(concat_image, caption="Target, Control & Generated")
                )
            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    logger.debug("Cleaning up pipeline")
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Validation complete.")
    return image_logs


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
    }


def main(config: ControlNetTrainConfig):
    logging_dir = Path(config.output_dir, config.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
    )
    logger.info("Starting main function with config: %s", config)

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        logger.info("MPS detected, disabling native AMP.")
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info("Accelerator state: %s", accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if config.seed is not None:
        logger.info("Setting random seed: %d", config.seed)
        set_seed(config.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if config.output_dir is not None:
            logger.info("Ensuring output directory exists: %s", config.output_dir)
            os.makedirs(config.output_dir, exist_ok=True)

    # Load scheduler and models
    logger.info("Loading noise scheduler from: %s", config.pretrained_model_name_or_path)
    noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")
    logger.info("Noise scheduler config: %s", noise_scheduler.config)
    logger.info("Loading VAE from: %s", config.pretrained_model_name_or_path)
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="vae", revision=config.revision, variant=config.variant
    )
    logger.info("Loading UNet from: %s", config.pretrained_model_name_or_path)
    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="unet", revision=config.revision, variant=config.variant
    )

    if config.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights from: %s", config.controlnet_model_name_or_path)
        controlnet = ControlNetModel.from_pretrained(config.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet.")
        controlnet = ControlNetModel.from_unet(unet)

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        logger.info("Registering custom save/load hooks for accelerator state.")
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.train()

    if config.enable_xformers_memory_efficient_attention:
        logger.info("Enabling xformers memory efficient attention.")
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            logger.error("xformers is not available. Make sure it is installed correctly")
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if config.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing for controlnet.")
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnet).dtype != torch.float32:
        logger.error(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        logger.info("Enabling TF32 for CUDA matmul.")
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.scale_lr:
        logger.info("Scaling learning rate by batch/accumulation/processes.")
        config.learning_rate = (
            config.learning_rate * config.gradient_accumulation_steps * config.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if config.use_8bit_adam:
        logger.info("Using 8-bit Adam optimizer.")
        try:
            import bitsandbytes as bnb
        except ImportError:
            logger.error("To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`.")
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        logger.info("Using standard AdamW optimizer.")
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    logger.info("Creating optimizer.")
    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    # Use RelightDataset instead of make_train_dataset
    logger.info("Loading training dataset from: %s", config.train_data_dir)
    train_dataset = RelightDataset(
        data_dir=config.train_data_dir,
        image_size=config.resolution,
        normalize_images=True
    )
    logger.info("Training dataset loaded with %d samples.", len(train_dataset))

    logger.info("Creating training dataloader.")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=config.train_batch_size,
        num_workers=config.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    logger.info("Setting up learning rate scheduler.")
    num_warmup_steps_for_scheduler = config.lr_warmup_steps * accelerator.num_processes
    if config.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / config.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            config.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = config.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=config.lr_num_cycles,
        power=config.lr_power,
    )

    logger.info("Preparing models, optimizer, dataloader, and scheduler with accelerator.")
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    logger.info("Moving VAE and UNet to device: %s, dtype: %s", accelerator.device, weight_dtype)
    # Move vae and unet to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # Create null text embeddings
    null_embeddings = torch.zeros((1, 77, 768), device=accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != config.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    config.num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        if config.report_to == "wandb":
            logger.info("Logging in to wandb.")
            wandb.login(key=get_wandb_key())
        tracker_config = dict(vars(config))
        logger.info("Initializing accelerator trackers.")
        accelerator.init_trackers(config.tracker_project_name, config=tracker_config)
        if config.report_to == "wandb":
            # Log the full config to wandb for experiment tracking
            wandb.config.update(dict(vars(config)), allow_val_change=True)

    # Train!
    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if config.resume_from_checkpoint:
        logger.info("Resuming from checkpoint: %s", config.resume_from_checkpoint)
        if config.resume_from_checkpoint != "latest":
            path = os.path.basename(config.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            logger.warning("Checkpoint '%s' does not exist. Starting a new training run.", config.resume_from_checkpoint)
            config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            logger.info("Loading state from checkpoint: %s", path)
            accelerator.load_state(os.path.join(config.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    logger.info("Creating progress bar for training.")
    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    for epoch in range(first_epoch, config.num_train_epochs):
        logger.info(f"Starting epoch {epoch+1}/{config.num_train_epochs}")
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents.float(), noise.float(), timesteps).to(
                    dtype=weight_dtype
                )

                # Use null embeddings for conditioning
                encoder_hidden_states = null_embeddings.repeat(bsz, 1, 1)

                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    logger.error(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=config.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % config.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if config.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(config.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= config.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(config.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if config.validation_data_dir is not None and global_step % config.validation_steps == 0:
                        logger.info(f"Running validation at step {global_step}")
                        image_logs = log_validation(
                            vae,
                            unet,
                            controlnet,
                            config,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= config.max_train_steps:
                logger.info("Reached max training steps. Ending training loop.")
                break

    # Create the pipeline using the trained modules and save it.
    logger.info("Waiting for all processes to finish before saving final model.")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Saving final controlnet model to %s", config.output_dir)
        controlnet = unwrap_model(controlnet)
        controlnet.save_pretrained(config.output_dir)

        # Run a final round of validation.
        image_logs = None
        if config.validation_data_dir is not None:
            logger.info("Running final validation after training.")
            image_logs = log_validation(
                vae=vae,
                unet=unet,
                controlnet=None,
                config=config,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=global_step,
                is_final_validation=True,
            )

    logger.info("Training complete. Exiting main function.")
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    config = ControlNetTrainConfig.from_args(args)
    main(config)