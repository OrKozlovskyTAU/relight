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

import logging
import math
import os
from pathlib import Path

import diffusers
import plotly.graph_objs as go
import plotly.io as pio
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from plotly.subplots import make_subplots
from tqdm.auto import tqdm

import wandb
from relight.cli.train import parse_args
from relight.trainers.flux_trainer import ControlNetFluxTrainer
from relight.trainers.sd3_trainer import ControlNetSD3Trainer
from relight.trainers.sd_trainer import ControlNetUnifiedTrainer
from relight.training.config import ControlNetTrainConfig
from relight.training.dataset import RelightDataset
from relight.utils.wandb_key import get_wandb_key

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.34.0.dev0")

logger = get_logger(__name__)

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

    # Restrict logging to main process
    if not accelerator.is_main_process:
        logging.getLogger().setLevel(logging.WARNING)

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

    # Model selection
    if config.model_type == "sd15" or config.model_type == "sd21":
        trainer = ControlNetUnifiedTrainer(config, logger, accelerator.device)
    elif config.model_type == "sd3":
        trainer = ControlNetSD3Trainer(config, logger, accelerator.device)
    elif config.model_type == "flux":
        trainer = ControlNetFluxTrainer(config, logger, accelerator.device)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    # Optimizer creation
    logger.info("Creating optimizer.")
    params_to_optimize = trainer.get_parameters_to_optimize()
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
        normalize_images=True,
        subset_size=config.subset_size,
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
        step_rules=config.step_rules,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=config.lr_num_cycles,
        power=config.lr_power,
    )

    # Prepare model submodules with accelerator using trainer's method
    trainer.prepare_models_for_accelerator(accelerator)
    # Prepare optimizer, dataloader, and scheduler separately
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

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

    # For tracking gradients and weights norms per layer per step
    grad_norms_per_layer = {}
    weight_norms_per_layer = {}
    steps_tracked = []

    # Validation logic using trainer abstraction
    def run_validation_with_trainer(trainer, config, accelerator, weight_dtype, step, is_final_validation=False):
        validation_dataset = RelightDataset(
            data_dir=config.validation_data_dir,
            image_size=config.resolution,
            normalize_images=False
        )
        pipeline = trainer.get_validation_pipeline(accelerator, weight_dtype, is_final_validation)
        trainer.run_validation(
            pipeline, validation_dataset, accelerator, step, is_final_validation
        )

    for epoch in range(first_epoch, config.num_train_epochs):
        logger.info(f"Starting epoch {epoch+1}/{config.num_train_epochs}")
        for batch in train_dataloader:
            with accelerator.accumulate(trainer.get_accumulate_object()):
                batch_data = trainer.prepare_batch(batch, accelerator.device, weight_dtype)
                outputs = trainer.forward(batch_data)
                loss = trainer.compute_loss(outputs)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = trainer.get_parameters_to_clip()
                    accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)
                
                # Log gradients and weights
                if (global_step % config.log_grad_and_weights_steps == 0 or global_step == 0) and config.report_to == "wandb":
                    total_grad_norm = 0.0
                    num_params = 0
                    for name, param in trainer.controlnet.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            total_grad_norm += abs(grad_norm)
                            # Track for plotting
                            if name not in grad_norms_per_layer:
                                grad_norms_per_layer[name] = []
                            grad_norms_per_layer[name].append(grad_norm)
                            if name not in weight_norms_per_layer:
                                weight_norms_per_layer[name] = []
                            weight_norms_per_layer[name].append(param.data.norm().item())
                            num_params += 1
                    steps_tracked.append(global_step)
                    if num_params > 0:
                        accelerator.log({
                            "gradients/total_gradients_norm": total_grad_norm / num_params
                        }, step=global_step)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=config.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if config.checkpointing_steps is not None and global_step % config.checkpointing_steps == 0:
                        save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    if config.validation_data_dir is not None and global_step % config.validation_steps == 0:
                        run_validation_with_trainer(
                            trainer, config, accelerator, weight_dtype, global_step, is_final_validation=False
                        )

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= config.max_train_steps:
                break
        
        if global_step >= config.max_train_steps:
            logger.info("Reached max training steps. Ending training loop.")
            break

    # --- PLOTLY 3D PLOTS AND WANDB LOGGING ---
    if accelerator.is_main_process and config.report_to == "wandb":

        def get_group_name(layer_name):
            parts = layer_name.split('.')
            if parts[0] == 'module':
                if parts[1].startswith('controlnet'):
                    return 'controlnet'
                elif parts[1] in ['conv_in', 'down_blocks', 'mid_block']:
                    return 'main_model'
                else:
                    return None  # Exclude other modules
            return None

        def plot_norms_by_group(norms_per_layer, steps, layer_names, title, html_path):
            # Filter out bias layers
            filtered_layer_names = [name for name in layer_names if not name.endswith('.bias')]
            groups = {'main_model': [], 'controlnet': []}
            for name in filtered_layer_names:
                group = get_group_name(name)
                if group in groups:
                    groups[group].append(name)
            # Only keep non-empty groups
            groups = {k: v for k, v in groups.items() if v}
            rows = 1
            cols = 2
            fig = make_subplots(
                rows=rows, cols=cols,
                specs=[[{'type': 'surface'}]*cols],
                subplot_titles=[k.replace('_', ' ').title() for k in groups.keys()]
            )
            for i, (group, names) in enumerate(groups.items()):
                row = 1
                col = i + 1
                norm_matrix = [norms_per_layer[n] for n in names]
                fig.add_trace(
                    go.Surface(z=norm_matrix, x=steps, y=names, showscale=False),
                    row=row, col=col
                )
                # Set axis titles for each subplot
                fig.update_scenes(
                    dict(
                        xaxis_title='Step',
                        yaxis_title='Layer',
                        zaxis_title='Norm',
                    ),
                    row=row, col=col
                )
            fig.update_layout(
                title=title,
            )
            pio.write_html(fig, html_path)
            accelerator.log({f"plots/{os.path.basename(html_path)}": wandb.Html(html_path)})
            logger.info(f"Saved plot to {html_path}")

        layer_names = list(grad_norms_per_layer.keys())
        steps = steps_tracked
        # Gradients norm plot by group
        grad_html_path = os.path.join(config.output_dir, 'gradients_norm.html')
        plot_norms_by_group(
            grad_norms_per_layer, steps, layer_names,
            'Gradients Norm per Layer per Step', grad_html_path
        )
        # Weights norm plot by group
        weight_html_path = os.path.join(config.output_dir, 'weights_norm.html')
        plot_norms_by_group(
            weight_norms_per_layer, steps, layer_names,
            'Weights Norm per Layer per Step', weight_html_path
        )

    # Create the pipeline using the trained modules and save it.
    logger.info("Waiting for all processes to finish before saving final model.")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if config.checkpointing_steps is not None:
            logger.info("Saving final model to %s", config.output_dir)
            trainer.save(config.output_dir)
        # Final validation
        if config.validation_data_dir is not None:
            run_validation_with_trainer(
                trainer, config, accelerator, weight_dtype, global_step, is_final_validation=True
            )

    logger.info("Training complete. Exiting main function.")
    accelerator.end_training()


if __name__ == "__main__":
    # To use with wandb sweeps, run:
    #   wandb sweep wandb_sweep.yaml
    #   wandb agent <sweep_id>
    # if 'wandb' in sys.modules:
    args = parse_args()
    config = ControlNetTrainConfig.from_args(args)

    if config.wandb_sweep_agent:
        wandb.login(key=get_wandb_key())
        wandb.init(project=config.tracker_project_name)
        config = ControlNetTrainConfig.from_wandb_config(wandb.config)

    main(config)