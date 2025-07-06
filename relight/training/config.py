from dataclasses import dataclass
from typing import Optional


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
    # If None, checkpoints will not be saved.
    checkpointing_steps: Optional[int] = None

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

    # Number of validation samples to use from the validation dataset. If None, use all.
    max_validation_samples: Optional[int] = None

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
    # Controls the  maximum number of denoising steps for validation samples.
    validation_num_inference_steps: int = 50

    # Loss weights for training
    
    # Weight for the Mean Squared Error (MSE) loss component. MSE loss measures the average squared difference
    # between predicted and target values, heavily penalizing large errors. Default is 1.0 to use MSE as the primary loss.
    mse_loss_weight: float = 1.0

    # Number of steps between logging gradients and weights to wandb during training
    log_grad_and_weights_steps: Optional[int] = None

    # Number of images to use from the training dataset.
    subset_size: Optional[int] = None

    # Factor for the zero-frequency extra noise added to latents during training.
    # This would make it so that the model learns to change the zero-frequency of the component freely,
    # randomized ~1/`noise_zero_frequency_factor` times faster than for the base distribution.
    noise_zero_frequency_factor: float = 0.1

    # Rules for step-based learning rate adjustments.
    # Format depends on the scheduler, e.g., for MultiStepLR, it's a list of epochs.
    # Example: "10,20,30"
    step_rules: Optional[str] = None

    # If True, apply LAB color matching to generated images before logging (validation and training logs)
    lab_color_match_logging: bool = False

    # If True, initialize config from wandb.config for sweep agent mode
    wandb_sweep_agent: bool = False

    model_type: str = "sd15"  # or "sd3", "flux"

    # --- Validation trade-off logic ---
    # Minimum percent improvement required in perceptual loss to consider as better 
    validation_perceptual_improve_pct: float = 10.0
    # Minimum percent improvement required in MSE loss to consider as better 
    validation_mse_improve_pct: float = 1.0

    @staticmethod
    def from_args(args) -> 'ControlNetTrainConfig':
        # Only keep keys that are fields of ControlNetTrainConfig
        valid_keys = set(ControlNetTrainConfig.__dataclass_fields__.keys())
        filtered_args = {k: v for k, v in vars(args).items() if k in valid_keys}
        return ControlNetTrainConfig(**filtered_args)

    @staticmethod
    def from_wandb_config(wandb_config) -> 'ControlNetTrainConfig':
        valid_keys = set(ControlNetTrainConfig.__dataclass_fields__.keys())
        filtered_args = {k: v for k, v in dict(wandb_config).items() if k in valid_keys}
        return ControlNetTrainConfig(**filtered_args) 
