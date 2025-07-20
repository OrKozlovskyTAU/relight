import os
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)

from relight.trainers.model_base import ControlNetTrainerBase
from relight.training.config import ControlNetTrainConfig
from typing import Any, Dict, Optional


class ControlNetUnifiedTrainer(ControlNetTrainerBase):
    def __init__(self, config: ControlNetTrainConfig, logger: Any, device: torch.device) -> None:
        logger.info(f"Loading vae (AutoencoderKL) from: {config.pretrained_model_name_or_path}")
        self.vae = AutoencoderKL.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="vae", revision=config.revision, variant=config.variant
        )
        logger.info(f"Loading unet (UNet2DConditionModel) from: {config.pretrained_model_name_or_path}")
        self.unet = UNet2DConditionModel.from_pretrained(
            config.pretrained_model_name_or_path, subfolder="unet", revision=config.revision, variant=config.variant
        )
        if config.controlnet_model_name_or_path:
            logger.info(f"Loading existing controlnet weights from: {config.controlnet_model_name_or_path}")
            self.controlnet = ControlNetModel.from_pretrained(config.controlnet_model_name_or_path)
        else:
            logger.info("Initializing controlnet weights from unet.")
            self.controlnet = ControlNetModel.from_unet(self.unet)
        logger.info(f"Loading noise scheduler (DDPMScheduler) from: {config.pretrained_model_name_or_path}")
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="scheduler",
            rescale_betas_zero_snr=config.rescale_betas_zero_snr,
            timestep_spacing=config.timestep_spacing,
        )
        logger.info(f"Noise scheduler config: {self.noise_scheduler.config}")
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.controlnet.train()
        logger.info("All models loaded successfully")
        super().__init__(config, logger, device)

    def prepare_batch(self, batch: Dict[str, Any], device: torch.device, weight_dtype: torch.dtype) -> Dict[str, Any]:
        pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
        conditioning_pixel_values = batch["conditioning_pixel_values"].to(device, dtype=weight_dtype)
        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
        }

    def forward(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        config = self.config
        latents = self.vae.encode(batch_data["pixel_values"]).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        noise = torch.randn_like(latents)
        if config.noise_zero_frequency_factor > 0:
            noise = noise + config.noise_zero_frequency_factor * torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=latents.device, dtype=latents.dtype)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents.float(), noise.float(), timesteps).to(dtype=latents.dtype)
        # Determine hidden size from unet config
        hidden_size = getattr(self.unet.config, 'cross_attention_dim', 768)  # SD15: 768, SD21: 1024
        encoder_hidden_states = torch.zeros((bsz, 77, hidden_size), device=latents.device, dtype=latents.dtype)
        controlnet_image = batch_data["conditioning_pixel_values"]
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_image,
            return_dict=False,
        )
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=[sample.to(dtype=latents.dtype) for sample in down_block_res_samples],
            mid_block_additional_residual=mid_block_res_sample.to(dtype=latents.dtype),
            return_dict=False,
        )[0]
        return {
            "model_pred": model_pred,
            "latents": latents,
            "noise": noise,
            "timesteps": timesteps,
            "noisy_latents": noisy_latents,
        }

    def compute_loss(self, outputs: Dict[str, Any]) -> torch.Tensor:
        # Use the prediction_type logic from the provided snippet
        prediction_type = getattr(self.noise_scheduler.config, 'prediction_type', 'epsilon')
        if prediction_type == "epsilon":
            target = outputs["noise"]
        elif prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(outputs["latents"], outputs["noise"], outputs["timesteps"])
        else:
            raise ValueError(f"Unknown prediction type {prediction_type}")
        mse_loss = F.mse_loss(outputs["model_pred"].float(), target.float(), reduction="mean")
        return mse_loss

    @property
    def controlnet_class(self) -> type:
        return ControlNetModel

    @property
    def pipeline_class(self) -> type:
        return StableDiffusionControlNetPipeline

    def get_validation_inputs(self, sample: Any, accelerator: Any) -> Dict[str, Any]:
        config = self.config
        control_image_path = os.path.join(config.validation_data_dir, sample['control_file'])
        target_image_path = os.path.join(config.validation_data_dir, sample['target_file'])
        control_image = Image.open(control_image_path).convert("RGB")
        target_image = Image.open(target_image_path).convert("RGB")
        return {
            'prompt': '',  # Null prompt
            'image': control_image,
            'target_image': target_image,
        }

    def prepare_models_for_accelerator(self, accelerator: Any) -> None:
        self.vae, self.unet, self.controlnet = accelerator.prepare(self.vae, self.unet, self.controlnet)

    def get_validation_pipeline(self, accelerator: Any, weight_dtype: torch.dtype, is_final_validation: bool) -> Any:
        controlnet = accelerator.unwrap_model(self.controlnet)
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.vae.config._name_or_path,
            vae=self.vae,
            unet=self.unet,
            controlnet=controlnet,
            safety_checker=None,
            revision=self.config.revision,
            variant=self.config.variant,
            torch_dtype=weight_dtype,
        )
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        return pipeline 