import os
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3ControlNetModel,
    SD3Transformer2DModel,
    StableDiffusion3ControlNetPipeline,
)
from diffusers.training_utils import compute_loss_weighting_for_sd3

from relight.trainers.model_base import ControlNetTrainerBase
from relight.training.config import ControlNetTrainConfig
from typing import Any, Dict, Optional


class ControlNetSD3Trainer(ControlNetTrainerBase):
    def __init__(self, config: ControlNetTrainConfig, logger: Any, device: torch.device) -> None:
        self.vae = AutoencoderKL.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="vae", revision=config.revision, variant=config.variant
        )
        self.transformer = SD3Transformer2DModel.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="transformer", revision=config.revision, variant=config.variant
        )
        if config.controlnet_model_name_or_path:
            logger.info("Loading existing controlnet weights from: %s", config.controlnet_model_name_or_path)
            self.controlnet = SD3ControlNetModel.from_pretrained(config.controlnet_model_name_or_path)
        else:
            logger.info("Initializing controlnet weights from transformer.")
            self.controlnet = SD3ControlNetModel.from_transformer(self.transformer)
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.noise_scheduler_copy = self.noise_scheduler
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.controlnet.train()
        super().__init__(config, logger, device)

    def prepare_batch(self, batch: Dict[str, Any], device: torch.device, weight_dtype: torch.dtype) -> Dict[str, Any]:
        pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
        conditioning_pixel_values = batch["conditioning_pixel_values"].to(device, dtype=weight_dtype)
        prompt_embeds = batch["prompt_embeds"].to(device, dtype=weight_dtype)
        pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(device, dtype=weight_dtype)
        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
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
        pixel_values = batch_data["pixel_values"]
        vae = self.vae
        model_input = vae.encode(pixel_values).latent_dist.sample()
        model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
        model_input = model_input.to(dtype=pixel_values.dtype)
        noise = torch.randn_like(model_input)
        if config.noise_zero_frequency_factor > 0:
            noise = noise + config.noise_zero_frequency_factor * torch.randn(model_input.shape[0], model_input.shape[1], 1, 1, device=model_input.device, dtype=model_input.dtype)
        bsz = model_input.shape[0]
        u = torch.rand(bsz, device=model_input.device)
        indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
        sigmas = self.get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype, device=model_input.device)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
        prompt_embeds = batch_data["prompt_embeds"]
        pooled_prompt_embeds = batch_data["pooled_prompt_embeds"]
        controlnet_image = batch_data["conditioning_pixel_values"]
        controlnet_image = vae.encode(controlnet_image).latent_dist.sample()
        controlnet_image = controlnet_image * vae.config.scaling_factor
        control_block_res_samples = self.controlnet(
            hidden_states=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            controlnet_cond=controlnet_image,
            return_dict=False,
        )[0]
        control_block_res_samples = [sample.to(dtype=model_input.dtype) for sample in control_block_res_samples]
        model_pred = self.transformer(
            hidden_states=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            block_controlnet_hidden_states=control_block_res_samples,
            return_dict=False,
        )[0]
        return {
            "model_pred": model_pred,
            "model_input": model_input,
            "noise": noise,
            "timesteps": timesteps,
            "sigmas": sigmas,
            "noisy_model_input": noisy_model_input,
        }

    def compute_loss(self, outputs: Dict[str, Any]) -> torch.Tensor:
        config = self.config
        model_pred = outputs["model_pred"]
        model_input = outputs["model_input"]
        noise = outputs["noise"]
        sigmas = outputs["sigmas"]
        noisy_model_input = outputs["noisy_model_input"]
        if config.precondition_outputs:
            target = model_input
            model_pred = model_pred * (-sigmas) + noisy_model_input
        else:
            target = noise - model_input
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=config.weighting_scheme, sigmas=sigmas)
        mse_loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        ).mean()
        return mse_loss

    @property
    def controlnet_class(self) -> type:
        return SD3ControlNetModel

    @property
    def pipeline_class(self) -> type:
        return StableDiffusion3ControlNetPipeline

    def get_validation_inputs(self, sample: Any, accelerator: Any) -> Dict[str, Any]:
        config = self.config
        control_image_path = os.path.join(config.validation_data_dir, sample['control_file'])
        target_image_path = os.path.join(config.validation_data_dir, sample['target_file'])
        # For SD3, use null embeddings for validation
        prompt_embeds = torch.zeros((1, 77, 768), dtype=torch.float32, device=accelerator.device)
        negative_prompt_embeds = torch.zeros((1, 77, 768), dtype=torch.float32, device=accelerator.device)
        pooled_prompt_embeds = torch.zeros((1, 768), dtype=torch.float32, device=accelerator.device)
        negative_pooled_prompt_embeds = torch.zeros((1, 768), dtype=torch.float32, device=accelerator.device)
        return {
            'prompt_embeds': prompt_embeds,
            'negative_prompt_embeds': negative_prompt_embeds,
            'pooled_prompt_embeds': pooled_prompt_embeds,
            'negative_pooled_prompt_embeds': negative_pooled_prompt_embeds,
            'control_image': control_image_path,
            'control_image_path': control_image_path,
            'target_image_path': target_image_path,
        }

    def prepare_models_for_accelerator(self, accelerator: Any) -> None:
        self.vae, self.transformer, self.controlnet = accelerator.prepare(self.vae, self.transformer, self.controlnet)
