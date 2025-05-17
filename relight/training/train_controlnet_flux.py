import argparse
import copy
import functools
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.logging import get_logger
from accelerate.utils import DistributedType
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm


import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.training_utils import compute_density_for_timestep_sampling, free_memory
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from relight.cli.train import parse_args
from relight.utils.training_utils import unwrap_model, save_model_card, get_sigmas, setup_accelerator, setup_logging, create_output_dir, save_checkpoint, validate_training_args, create_model_hooks, load_models

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.34.0.dev0")

logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def log_validation(
    vae, flux_transformer, flux_controlnet, args, accelerator, weight_dtype, step, is_final_validation=False
):
    logger.info("Running validation... ")

    if not is_final_validation:
        flux_controlnet = unwrap_model(accelerator, flux_controlnet)
        pipeline = FluxControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            controlnet=flux_controlnet,
            transformer=flux_transformer,
            torch_dtype=torch.bfloat16,
        )
    else:
        flux_controlnet = FluxControlNetModel.from_pretrained(
            args.output_dir, torch_dtype=torch.bfloat16, variant=args.save_weight_dtype
        )
        pipeline = FluxControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            controlnet=flux_controlnet,
            transformer=flux_transformer,
            torch_dtype=torch.bfloat16,
        )

    pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    validation_data_dirs = args.validation_data_dir if isinstance(args.validation_data_dir, list) else [args.validation_data_dir]
    image_logs = []
    if is_final_validation or torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    for validation_data_dir in validation_data_dirs:
        from diffusers.utils import load_image

        validation_image = load_image(validation_data_dir)
        validation_image = validation_image.resize((args.resolution, args.resolution))

        images = []
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            "", prompt_2=""
        )
        for _ in range(args.num_validation_images):
            with autocast_ctx:
                image = pipeline(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    control_image=validation_image,
                    num_inference_steps=28,
                    controlnet_conditioning_scale=1,
                    guidance_scale=3.5,
                    generator=generator,
                ).images[0]
            image = image.resize((args.resolution, args.resolution))
            images.append(image)
        image_logs.append(
            {"validation_data_dir": validation_image, "images": images}
        )

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_data_dir = log["validation_data_dir"]

                formatted_images = [np.asarray(validation_data_dir)]

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images("validation", formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_data_dir = log["validation_data_dir"]

                formatted_images.append(wandb.Image(validation_data_dir, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(image, caption="validation")
                    formatted_images.append(image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

        del pipeline
        free_memory()
        return image_logs


def get_train_dataset(args, accelerator):
    dataset = None
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    if args.jsonl_for_train is not None:
        # load from json
        dataset = load_dataset("json", data_files=args.jsonl_for_train, cache_dir=args.cache_dir)
        dataset = dataset.flatten_indices()
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    with accelerator.main_process_first():
        train_dataset = dataset["train"].shuffle(seed=args.seed)
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
    return train_dataset


def prepare_train_dataset(dataset, accelerator):
    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [
            (image.convert("RGB") if not isinstance(image, str) else Image.open(image).convert("RGB"))
            for image in examples[args.image_column]
        ]
        images = [image_transforms(image) for image in images]

        conditioning_images = [
            (image.convert("RGB") if not isinstance(image, str) else Image.open(image).convert("RGB"))
            for image in examples[args.conditioning_image_column]
        ]
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]
        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images

        return examples

    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess_train)

    return dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    prompt_ids = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])

    pooled_prompt_embeds = torch.stack([torch.tensor(example["pooled_prompt_embeds"]) for example in examples])
    text_ids = torch.stack([torch.tensor(example["text_ids"]) for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "prompt_ids": prompt_ids,
        "unet_added_conditions": {"pooled_prompt_embeds": pooled_prompt_embeds, "time_ids": text_ids},
    }


def main(args):
    # Validate training arguments
    validate_training_args(args)
    
    # Set up accelerator
    accelerator = setup_accelerator(args)
    
    # Set up logging
    logger = setup_logging(accelerator, args)
    
    # Create output directory and handle repository creation
    repo_id = create_output_dir(args, accelerator)
    
    # Load models
    models, (noise_scheduler, noise_scheduler_copy) = load_models(args, logger, model_type="flux")
    vae, transformer, controlnet = models

    weight_dtype = setup_weight_dtype(args, accelerator)

    # Setup training
    setup_training(args, accelerator, transformer, vae, controlnet, weight_dtype, model_type="flux")

    # Create model hooks for saving and loading
    create_model_hooks(accelerator, args, [controlnet], model_type="flux")

    # use some pipeline function
    flux_controlnet_pipeline = FluxControlNetPipeline(
        scheduler=noise_scheduler,
        vae=vae,
        text_encoder=text_encoder_one,
        tokenizer=tokenizer_one,
        text_encoder_2=text_encoder_two,
        tokenizer_2=tokenizer_two,
        transformer=transformer,
        controlnet=controlnet,
    )
    if args.enable_model_cpu_offload:
        flux_controlnet_pipeline.enable_model_cpu_offload()
    else:
        flux_controlnet_pipeline.to(accelerator.device)

    optimizer = setup_optimizer(args, controlnet)

    train_dataset = get_train_dataset(args, accelerator)

    compute_embeddings_fn = functools.partial(
        compute_text_embeddings,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        device=accelerator.device
    )
    with accelerator.main_process_first():
        from datasets.fingerprint import Hasher

        # fingerprint used by the cache for the other processes to load the result
        # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401
        new_fingerprint = Hasher.hash(args)
        train_dataset = train_dataset.map(
            compute_embeddings_fn,
            batched=True,
            batch_size=args.dataset_preprocess_batch_size,
            new_fingerprint=new_fingerprint,
        )

    del text_encoder_one, text_encoder_two
    del tokenizer_one, tokenizer_two
    free_memory()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        # tracker_config.pop("validation_prompt")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        return get_sigmas(timesteps, noise_scheduler_copy, n_dim=n_dim, dtype=dtype, device=accelerator.device)

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                # vae encode
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                pixel_latents_tmp = vae.encode(pixel_values).latent_dist.sample()
                pixel_latents_tmp = (pixel_latents_tmp - vae.config.shift_factor) * vae.config.scaling_factor
                pixel_latents = FluxControlNetPipeline._pack_latents(
                    pixel_latents_tmp,
                    pixel_values.shape[0],
                    pixel_latents_tmp.shape[1],
                    pixel_latents_tmp.shape[2],
                    pixel_latents_tmp.shape[3],
                )

                control_values = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                control_latents = vae.encode(control_values).latent_dist.sample()
                control_latents = (control_latents - vae.config.shift_factor) * vae.config.scaling_factor
                control_image = FluxControlNetPipeline._pack_latents(
                    control_latents,
                    control_values.shape[0],
                    control_latents.shape[1],
                    control_latents.shape[2],
                    control_latents.shape[3],
                )

                latent_image_ids = FluxControlNetPipeline._prepare_latent_image_ids(
                    batch_size=pixel_latents_tmp.shape[0],
                    height=pixel_latents_tmp.shape[2] // 2,
                    width=pixel_latents_tmp.shape[3] // 2,
                    device=pixel_values.device,
                    dtype=pixel_values.dtype,
                )

                bsz = pixel_latents.shape[0]
                noise = torch.randn_like(pixel_latents).to(accelerator.device).to(dtype=weight_dtype)
                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)

                # Add noise according to flow matching.
                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

                # handle guidance
                if transformer.config.guidance_embeds:
                    guidance_vec = torch.full(
                        (noisy_model_input.shape[0],),
                        args.guidance_scale,
                        device=noisy_model_input.device,
                        dtype=weight_dtype,
                    )
                else:
                    guidance_vec = None

                controlnet_block_samples, controlnet_single_block_samples = controlnet(
                    hidden_states=noisy_model_input,
                    controlnet_cond=control_image,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=batch["unet_added_conditions"]["pooled_prompt_embeds"].to(dtype=weight_dtype),
                    encoder_hidden_states=batch["prompt_ids"].to(dtype=weight_dtype),
                    txt_ids=batch["unet_added_conditions"]["time_ids"][0].to(dtype=weight_dtype),
                    img_ids=latent_image_ids,
                    return_dict=False,
                )

                noise_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=batch["unet_added_conditions"]["pooled_prompt_embeds"].to(dtype=weight_dtype),
                    encoder_hidden_states=batch["prompt_ids"].to(dtype=weight_dtype),
                    controlnet_block_samples=[sample.to(dtype=weight_dtype) for sample in controlnet_block_samples]
                    if controlnet_block_samples is not None
                    else None,
                    controlnet_single_block_samples=[
                        sample.to(dtype=weight_dtype) for sample in controlnet_single_block_samples
                    ]
                    if controlnet_single_block_samples is not None
                    else None,
                    txt_ids=batch["unet_added_conditions"]["time_ids"][0].to(dtype=weight_dtype),
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                loss = F.mse_loss(noise_pred.float(), (noise - pixel_latents).float(), reduction="mean")
                accelerator.backward(loss)
                # Check if the gradient of each model parameter contains NaN
                for name, param in controlnet.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        logger.error(f"Gradient for {name} contains NaN!")

                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
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

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        pass
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = unwrap_model(controlnet)
        save_weight_dtype = torch.float32
        if args.save_weight_dtype == "fp16":
            save_weight_dtype = torch.float16
        elif args.save_weight_dtype == "bf16":
            save_weight_dtype = torch.bfloat16
        controlnet.to(save_weight_dtype)
        if args.save_weight_dtype != "fp32":
            controlnet.save_pretrained(args.output_dir, variant=args.save_weight_dtype)
        else:
            controlnet.save_pretrained(args.output_dir)
        # Run a final round of validation.
        # Setting `vae`, `unet`, and `controlnet` to None to load automatically from `args.output_dir`.
        image_logs = None
        if args.validation_prompt is not None:
            pass

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)