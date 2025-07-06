import os

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline

from relight.trainers.model_base import ControlNetTrainerBase
from relight.training.config import ControlNetTrainConfig
from typing import Any, Dict, Optional

class ControlNetFluxTrainer(ControlNetTrainerBase):
    def __init__(self, config: ControlNetTrainConfig, logger: Any, device: torch.device) -> None:
        logger.info(f"Loading vae from: {config.pretrained_model_name_or_path}")
        self.vae = AutoencoderKL.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="vae", revision=config.revision, variant=config.variant
        )
        logger.info(f"Loading transformer from: {config.pretrained_model_name_or_path}")
        self.transformer = FluxTransformer2DModel.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="transformer", revision=config.revision, variant=config.variant
        )
        if config.controlnet_model_name_or_path:
            logger.info("Loading existing controlnet weights from: %s", config.controlnet_model_name_or_path)
            self.controlnet = FluxControlNetModel.from_pretrained(config.controlnet_model_name_or_path)
        else:
            logger.info("Initializing controlnet weights from transformer.")
            self.controlnet = FluxControlNetModel.from_transformer(
                self.transformer,
                attention_head_dim=self.transformer.config["attention_head_dim"],
                num_attention_heads=self.transformer.config["num_attention_heads"],
                num_layers=config.num_double_layers,
                num_single_layers=config.num_single_layers,
            )
        logger.info(f"Loading noise scheduler from: {config.pretrained_model_name_or_path}")
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.noise_scheduler_copy = self.noise_scheduler
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.controlnet.train()
        logger.info("All models loaded successfully")
        super().__init__(config, logger, device)

    def prepare_batch(self, batch: Dict[str, Any], device: torch.device, weight_dtype: torch.dtype) -> Dict[str, Any]:
        pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
        conditioning_pixel_values = batch["conditioning_pixel_values"].to(device, dtype=weight_dtype)
        prompt_ids = batch["prompt_ids"].to(device, dtype=weight_dtype)
        pooled_prompt_embeds = batch["unet_added_conditions"]["pooled_prompt_embeds"].to(device, dtype=weight_dtype)
        text_ids = batch["unet_added_conditions"]["time_ids"].to(device, dtype=weight_dtype)
        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "prompt_ids": prompt_ids,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "text_ids": text_ids,
        }

    def get_sigmas(self, timesteps: torch.Tensor, n_dim: int = 4, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
        sigmas = self.noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def forward(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        config = self.config
        vae = self.vae
        pixel_values = batch_data["pixel_values"]
        pixel_latents_tmp = vae.encode(pixel_values).latent_dist.sample()
        pixel_latents_tmp = (pixel_latents_tmp - vae.config.shift_factor) * vae.config.scaling_factor
        pixel_latents = pixel_latents_tmp
        control_values = batch_data["conditioning_pixel_values"]
        control_latents = vae.encode(control_values).latent_dist.sample()
        control_latents = (control_latents - vae.config.shift_factor) * vae.config.scaling_factor
        control_image = control_latents
        bsz = pixel_latents.shape[0]
        noise = torch.randn_like(pixel_latents).to(pixel_latents.device).to(dtype=pixel_latents.dtype)
        if config.noise_zero_frequency_factor > 0:
            noise = noise + config.noise_zero_frequency_factor * torch.randn(pixel_latents.shape[0], pixel_latents.shape[1], 1, 1, device=pixel_latents.device, dtype=pixel_latents.dtype)
        u = torch.rand(bsz, device=pixel_latents.device)
        indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)
        sigmas = self.get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype, device=pixel_latents.device)
        noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
        prompt_ids = batch_data["prompt_ids"]
        pooled_prompt_embeds = batch_data["pooled_prompt_embeds"]
        text_ids = batch_data["text_ids"]
        controlnet_block_samples, controlnet_single_block_samples = self.controlnet(
            hidden_states=noisy_model_input,
            controlnet_cond=control_image,
            timestep=timesteps / 1000,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_ids,
            txt_ids=text_ids[0],
            img_ids=None,
            return_dict=False,
        )
        noise_pred = self.transformer(
            hidden_states=noisy_model_input,
            timestep=timesteps / 1000,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_ids,
            controlnet_block_samples=[sample.to(dtype=noisy_model_input.dtype) for sample in controlnet_block_samples]
            if controlnet_block_samples is not None else None,
            controlnet_single_block_samples=[sample.to(dtype=noisy_model_input.dtype) for sample in controlnet_single_block_samples]
            if controlnet_single_block_samples is not None else None,
            txt_ids=text_ids[0],
            img_ids=None,
            return_dict=False,
        )[0]
        return {
            "noise_pred": noise_pred,
            "pixel_latents": pixel_latents,
            "noise": noise,
            "noisy_model_input": noisy_model_input,
            "timesteps": timesteps,
        }

    def compute_loss(self, outputs: Dict[str, Any]) -> torch.Tensor:
        noise_pred = outputs["noise_pred"]
        pixel_latents = outputs["pixel_latents"]
        noise = outputs["noise"]
        mse_loss = F.mse_loss(noise_pred.float(), (noise - pixel_latents).float(), reduction="mean")
        return mse_loss

    @property
    def controlnet_class(self) -> type:
        return FluxControlNetModel

    @property
    def pipeline_class(self) -> type:
        return FluxControlNetPipeline

    def get_validation_inputs(self, sample: Any, accelerator: Any) -> Dict[str, Any]:
        config = self.config
        control_image_path = os.path.join(config.validation_data_dir, sample['control_file'])
        target_image_path = os.path.join(config.validation_data_dir, sample['target_file'])
        prompt_embeds = torch.zeros((1, 77, 768), dtype=torch.float32, device=accelerator.device)
        pooled_prompt_embeds = torch.zeros((1, 768), dtype=torch.float32, device=accelerator.device)
        return {
            'prompt_embeds': prompt_embeds,
            'pooled_prompt_embeds': pooled_prompt_embeds,
            'control_image': control_image_path,
            'control_image_path': control_image_path,
            'target_image_path': target_image_path,
        }

    def prepare_models_for_accelerator(self, accelerator: Any) -> None:
        self.vae, self.transformer, self.controlnet = accelerator.prepare(self.vae, self.transformer, self.controlnet) 