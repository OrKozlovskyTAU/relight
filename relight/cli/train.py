import argparse
import os

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training script for ControlNet models.")
    
    # Common model arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    
    # Common training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", 
        type=int, 
        default=4, 
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int, 
        default=500, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power", 
        type=float, 
        default=1.0, 
        help="Power factor of the polynomial scheduler."
    )
    
    # ControlNet specific arguments
    parser.add_argument(
        "--num_extra_conditioning_channels",
        type=int,
        default=0,
        help="Number of extra conditioning channels for controlnet.",
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=("Coordinate for (the width) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--upcast_vae",
        action="store_true",
        help="Whether or not to upcast vae to fp32",
    )
    
    # Additional training configuration
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            "flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--enable_npu_flash_attention",
        action="store_true",
        help="Whether or not to use npu flash attention.",
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", 
        type=str, 
        default="image",
        help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_data_dir",
        type=str,
        default=None,
        help=(
            "Path to a folder containing controlnet conditioning images to be evaluated every `--validation_steps` "
            "and logged to `--report_to`. The folder should contain images matching the expected naming pattern."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to generate per validation sample during evaluation.",
    )
    parser.add_argument(
        "--max_validation_samples",
        type=int,
        default=None,
        help="Maximum number of validation samples to use from the validation dataset. If not set, use all.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run fine-tuning validation every X epochs. The validation process of the `train.py` script does not include DALL-E 2"
            " evaluation. If `--validation_epochs` is not set, the validation will not be run."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train-controlnet",
        help=(
            "The `project_name` argument passed to `accelerate.init_trackers` for more information on"
            " and all available options."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. If `"none"` is passed, no integrations will be used.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Used for distributed training.",
    )
    
    # Additional arguments from train_controlnet.py
    parser.add_argument(
        "--use_8bit_adam", 
        action="store_true", 
        help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--use_adafactor",
        action="store_true",
        help=(
            "Adafactor is a stochastic optimization method based on Adam that reduces memory usage while retaining"
            "the empirical benefits of adaptivity. This is achieved through maintaining a factored representation "
            "of the squared gradient accumulator across training steps."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help="Weighting scheme for the model.",
    )
    parser.add_argument(
        "--logit_mean", 
        type=float, 
        default=0.0, 
        help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", 
        type=float, 
        default=1.0, 
        help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--precondition_outputs",
        type=int,
        default=1,
        help="Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how "
        "model `target` is calculated.",
    )
    parser.add_argument(
        "--adam_beta1", 
        type=float, 
        default=0.9, 
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2", 
        type=float, 
        default=0.999, 
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay", 
        type=float, 
        default=1e-2, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon", 
        type=float, 
        default=1e-08, 
        help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument(
        "--max_grad_norm", 
        default=1.0, 
        type=float, 
        help="Max gradient norm."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    
    # Additional arguments from train_controlnet_flux.py
    parser.add_argument(
        "--num_double_layers",
        type=int,
        default=4,
        help="Number of double layers in the controlnet (default: 4).",
    )
    parser.add_argument(
        "--num_single_layers",
        type=int,
        default=4,
        help="Number of single layers in the controlnet (default: 4).",
    )
    parser.add_argument(
        "--jsonl_for_train",
        type=str,
        default=None,
        help="Path to the jsonl file containing the training data.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the guidance scale used for transformer.",
    )
    parser.add_argument(
        "--save_weight_dtype",
        type=str,
        default="fp32",
        choices=[
            "fp16",
            "bf16",
            "fp32",
        ],
        help=("Preserve precision type according to selected weight"),
    )
    parser.add_argument(
        "--enable_model_cpu_offload",
        action="store_true",
        help="Enable model cpu offload and save memory.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--dataset_preprocess_batch_size", 
        type=int, 
        default=1000, 
        help="Batch size for preprocessing dataset."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="sd3",
        choices=["sd3", "flux", "sd"],
        help="Type of ControlNet model to train: 'sd3' for standard ControlNet, 'flux' for Flux ControlNet, or 'sd' for ControlNet with null text embeddings.",
    )
    parser.add_argument(
        "--validation_num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps to use during validation image generation.",
    )

    # Loss weights for training
    parser.add_argument(
        "--mse_loss_weight",
        type=float,
        default=1.0,
        help="Weight for the MSE loss component.",
    )
    parser.add_argument(
        "--mae_loss_weight",
        type=float,
        default=0.0,
        help="Weight for the MAE loss component.",
    )
    parser.add_argument(
        "--perceptual_loss_weight",
        type=float,
        default=0.0,
        help="Weight for the perceptual (VGG) loss component.",
    )

    # Additional arguments from train_controlnet_flux.py
    parser.add_argument(
        "--log_training_image_steps",
        type=int,
        default=1000,
        help="Number of steps between logging generated/gt image pairs to wandb during training.",
    )
    parser.add_argument(
        "--log_grad_and_weights_steps",
        type=int,
        default=100,
        help="Number of steps between logging gradients and weights to wandb during training.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # Validation logic
    if args.dataset_name is None and args.train_data_dir is None and args.jsonl_for_train is None:
        raise ValueError("Specify either `--dataset_name`, `--train_data_dir`, or `--jsonl_for_train`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")
        
    if args.dataset_name is not None and args.jsonl_for_train is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--jsonl_for_train`")
        
    if args.train_data_dir is not None and args.jsonl_for_train is not None:
        raise ValueError("Specify only one of `--train_data_dir` or `--jsonl_for_train`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_data_dir is not None:
        if not os.path.isdir(args.validation_data_dir):
            raise ValueError(f"`--validation_data_dir` must be a path to a folder, got {args.validation_data_dir}")

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args 

def main():
    """Main function to run the training script."""
    args = parse_args()
    
    # Import the appropriate training module based on model_type
    if args.model_type == "sd3":
        from relight.training.train_controlnet_sd3 import main as train_controlnet
        train_controlnet(args)
    elif args.model_type == "flux":
        from relight.training.train_controlnet_flux import main as train_flux
        train_flux(args)
    elif args.model_type == "sd":
        from relight.training.train_controlnet import main, ControlNetTrainConfig
        config = ControlNetTrainConfig.from_args(args)
        main(config)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

if __name__ == "__main__":
    main() 