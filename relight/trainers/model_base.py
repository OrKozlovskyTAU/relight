import gc
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
import wandb
from torchvision.models import vgg16, VGG16_Weights

from relight.training.config import ControlNetTrainConfig


class ControlNetTrainerBase(ABC):
    @abstractmethod
    def __init__(self, config: ControlNetTrainConfig, logger: Any, device: torch.device) -> None:
        self.config = config
        self.logger = logger
        # Perceptual loss (VGG16) setup
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.to(device)
        self.vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.resize = transforms.Resize((224, 224))
        # Best validation tracking
        self.best_validation_metrics = None
        self.best_validation_step = None

    @abstractmethod
    def prepare_batch(self, batch: Dict[str, Any], device: torch.device, weight_dtype: torch.dtype) -> Dict[str, Any]:
        """
        Prepare and move batch data to device, return a dict of tensors needed for forward/loss.
        """

    @abstractmethod
    def forward(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the forward pass. Return model outputs needed for loss.
        """

    @abstractmethod
    def compute_loss(self, outputs: Dict[str, Any], batch_data: Dict[str, Any]) -> torch.Tensor:
        """
        Compute the loss given model outputs and batch data.
        """

    def save(self, output_dir: str) -> None:
        """
        Save the model to output_dir.
        """
        if hasattr(self, "controlnet") and self.controlnet is not None:
            self.controlnet.save_pretrained(output_dir)
        else:
            raise NotImplementedError("No controlnet model to save in base class.")

    @property
    @abstractmethod
    def controlnet_class(self) -> Type[Any]:
        """Return the class to use for loading the controlnet model."""

    @property
    @abstractmethod
    def pipeline_class(self) -> Type[Any]:
        """Return the class to use for the validation pipeline."""

    def get_pipeline_kwargs(self, accelerator: Any, weight_dtype: torch.dtype, is_final_validation: bool) -> Dict[str, Any]:
        """Override in subclass if you need to provide extra kwargs to the pipeline constructor."""
        return {}

    def load_for_validation(self, output_dir: str, weight_dtype: torch.dtype) -> None:
        self.controlnet = self.controlnet_class.from_pretrained(output_dir, torch_dtype=weight_dtype)

    def get_validation_pipeline(self, accelerator: Any, weight_dtype: torch.dtype, is_final_validation: bool) -> Any:
        # TODO: fix this - requires controlnet to be saved during training, which is not the case for now
        # if is_final_validation:
        #     controlnet = self.controlnet_class.from_pretrained(self.controlnet.module.config._name_or_path, torch_dtype=weight_dtype)
        # else:
        controlnet = accelerator.unwrap_model(self.controlnet)
        pipeline_kwargs = self.get_pipeline_kwargs(accelerator, weight_dtype, is_final_validation)
        pipeline = self.pipeline_class.from_pretrained(
            self.vae.config._name_or_path,
            controlnet=controlnet,
            torch_dtype=weight_dtype,
            safety_checker=None,
            guidance_rescale=self.config.guidance_rescale,
            **pipeline_kwargs
        )
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        return pipeline

    def setup_training(self, accelerator: Any) -> None:
        """
        Generic training setup for all trainers. Applies config-driven options:
        - scale_lr: scales learning rate by batch size, processes, and accumulation steps
        - allow_tf32: enables TF32 on CUDA if requested
        - enable_xformers_memory_efficient_attention: enables xformers on all submodules that support it
        - gradient_checkpointing: enables gradient checkpointing on all submodules that support it
        """
        config = self.config
        # Scale learning rate if requested
        if config.scale_lr:
            old_lr = config.learning_rate
            config.learning_rate = (
                config.learning_rate * config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
            )
            self.logger.info(f"Scaled learning rate from {old_lr} to {config.learning_rate}")
        # Enable TF32 if requested and available
        if config.allow_tf32:
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                self.logger.info("Enabled TF32 for CUDA matmul and cuDNN.")
        # Enable xformers memory efficient attention if requested
        if config.enable_xformers_memory_efficient_attention:
            for attr in ["unet", "controlnet", "transformer"]:
                module = getattr(self, attr, None)
                if module is not None and hasattr(module, "enable_xformers_memory_efficient_attention"):
                    try:
                        module.enable_xformers_memory_efficient_attention()
                        self.logger.info(f"Enabled xformers memory efficient attention on {attr}.")
                    except Exception as e:
                        self.logger.warning(f"Could not enable xformers on {attr}: {e}")
        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            for attr in ["unet", "controlnet", "transformer"]:
                module = getattr(self, attr, None)
                if module is not None and hasattr(module, "enable_gradient_checkpointing"):
                    module.enable_gradient_checkpointing()
                    self.logger.info(f"Enabled gradient checkpointing on {attr}.")

    def perceptual_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: [B, 3, H, W], range [0, 1]
        x = self.resize(x)
        y = self.resize(y)
        x = (x - self.vgg_mean) / self.vgg_std
        y = (y - self.vgg_mean) / self.vgg_std
        feat_x = self.vgg(x)
        feat_y = self.vgg(y)
        return F.l1_loss(feat_x, feat_y)

    @abstractmethod
    def get_validation_inputs(self, sample: Any, accelerator: Any) -> Dict[str, Any]:
        """
        Prepare the inputs for the pipeline call for a single validation sample.
        Should return a dict of kwargs to pass to the pipeline.
        """

    def is_better_validation(self, new_metrics: Dict[str, float], best_metrics: Optional[Dict[str, float]], x_pct: float = 1.0, y_pct: float = 1.0) -> bool:
        """
        Returns True if new_metrics is better than best_metrics according to the trade-off rule:
        - MSE loss must improve by at least y_pct percent (decrease),
        - Perceptual loss can degrade by at most x_pct percent (increase).
        """
        if best_metrics is None:
            return True
        perceptual_new = new_metrics["perceptual"]
        perceptual_best = best_metrics["perceptual"]
        mse_new = new_metrics["mse"]
        mse_best = best_metrics["mse"]
        perceptual_improvement = (perceptual_best - perceptual_new) / perceptual_best * 100
        mse_improvement = (mse_best - mse_new) / mse_best * 100
        perceptual_degradation = (perceptual_new - perceptual_best) / perceptual_best * 100
        mse_degradation = (mse_new - mse_best) / mse_best * 100
        if (mse_improvement >= y_pct and perceptual_degradation <= x_pct) or (perceptual_improvement >= x_pct and mse_degradation <= y_pct):    
            return True
        return False

    def run_validation(self, pipeline: Any, validation_dataset: Any, accelerator: Any, step: int, is_final_validation: bool) -> Any:
        config = self.config
        logger = self.logger
        logger.info("Running validation... ")
        if config.enable_xformers_memory_efficient_attention:
            pipeline.enable_xformers_memory_efficient_attention()
        if config.seed is None:
            generator = None
        else:
            generator = torch.Generator(device=accelerator.device).manual_seed(config.seed)
        indices = list(range(len(validation_dataset)))
        if config.seed is not None:
            g = torch.Generator()
            g.manual_seed(config.seed)
            indices = torch.randperm(len(validation_dataset), generator=g).tolist()
        else:
            random.shuffle(indices)
        image_logs = []
        
        inference_ctx = torch.autocast("cuda") if not is_final_validation else torch.no_grad()
        mae_losses = []
        perceptual_losses = []
        mse_losses = []
        max_samples = config.max_validation_samples if config.max_validation_samples is not None else len(validation_dataset)
        for idx in range(max_samples):
            sample = validation_dataset[indices[idx]]
            pipeline_inputs = self.get_validation_inputs(sample, accelerator)
            pipeline_inputs["generator"] = generator
            
            images = []
            steps_range = np.linspace(20, config.validation_num_inference_steps, config.num_validation_images, dtype=int)
            for _, num_steps in enumerate(steps_range):
                with inference_ctx:
                    pipeline_inputs["num_inference_steps"] = num_steps
                    image = pipeline(**pipeline_inputs).images[0]
                images.append(image)
            combined_images = [pipeline_inputs["image"], pipeline_inputs["target_image"]] + images
            image_logs.append({"images": combined_images})
            target_tensor = T.ToTensor()(pipeline_inputs["target_image"]).unsqueeze(0).to(accelerator.device)
            target_tensor = target_tensor.float()
            sample_mae_losses = []
            sample_perceptual_losses = []
            sample_mse_losses = []
            for image in images:
                gen_tensor = T.ToTensor()(image).unsqueeze(0).to(accelerator.device)
                gen_tensor = gen_tensor.float()
                mae = F.l1_loss(gen_tensor, target_tensor).item()
                perceptual = self.perceptual_loss(gen_tensor, target_tensor).item()
                mse = F.mse_loss(gen_tensor, target_tensor).item()
                sample_mae_losses.append(mae)
                sample_perceptual_losses.append(perceptual)
                sample_mse_losses.append(mse)
            mae_losses.append(min(sample_mae_losses))
            perceptual_losses.append(min(sample_perceptual_losses))
            mse_losses.append(min(sample_mse_losses))
        tracker_key = "test" if is_final_validation else "validation"
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                for log in image_logs:
                    images = log["images"]
                    formatted_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images("validation", formatted_images, step, dataformats="NHWC")
                tracker.writer.add_scalar(f"{tracker_key}/mae_loss", np.mean(mae_losses), step)
                tracker.writer.add_scalar(f"{tracker_key}/perceptual_loss", np.mean(perceptual_losses), step)
                tracker.writer.add_scalar(f"{tracker_key}/mse_loss", np.mean(mse_losses), step)
            elif tracker.name == "wandb":
                formatted_images = []
                for log in image_logs:
                    images = log["images"]
                    captions = ["Control", "Target"]
                    steps_range = np.linspace(20, config.validation_num_inference_steps, config.num_validation_images, dtype=int)
                    captions.extend([f"Generated | {int(steps)} steps" for steps in steps_range])
                    captioned_images = []
                    for img, caption in zip(images, captions):
                        img_array = np.asarray(img)
                        caption_space = np.ones((50, img_array.shape[1], 3), dtype=np.uint8) * 255
                        captioned_img = np.concatenate([img_array, caption_space], axis=0)
                        captioned_img = Image.fromarray(captioned_img)
                        draw = ImageDraw.Draw(captioned_img)
                        try:
                            font = ImageFont.truetype("DejaVuSans", 20)
                        except Exception:
                            font = None
                        draw.text((10, img_array.shape[0] + 5), caption, font=font, fill=(0, 0, 0))
                        captioned_images.append(np.array(captioned_img))
                    concat_image = np.concatenate(captioned_images, axis=1)
                    formatted_images.append(wandb.Image(concat_image))
                tracker.log({tracker_key: formatted_images, tracker_key + "/mae_loss": np.mean(mae_losses), tracker_key + "/perceptual_loss": np.mean(perceptual_losses), tracker_key + "/mse_loss": np.mean(mse_losses)})
            else:
                logger.warning(f"image logging not implemented for {tracker.name}")
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        # Track best validation score
        current_metrics = dict(mae=np.mean(mae_losses), perceptual=np.mean(perceptual_losses), mse=np.mean(mse_losses))
        if self.is_better_validation(current_metrics, self.best_validation_metrics, config.validation_perceptual_improve_pct, config.validation_mse_improve_pct):
            self.best_validation_metrics = current_metrics
            self.best_validation_step = step
            tracker.log({f"{tracker_key}/best_perceptual_loss": current_metrics["perceptual"], f"{tracker_key}/best_mse_loss": current_metrics["mse"]})
            logger.info(f"New best validation score at step {step}: {current_metrics}")
        return image_logs, current_metrics

    def get_parameters_to_optimize(self) -> Any:
        return self.controlnet.parameters()

    def get_accumulate_object(self) -> Any:
        return self

    def get_parameters_to_clip(self) -> Any:
        return self.controlnet.parameters() 
