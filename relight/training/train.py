"""
Common training script for ControlNet models.
"""

import os
import argparse
import logging
import importlib
import sys
from pathlib import Path

import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed

from relight.training.dataset import get_train_dataset
from relight.training.train_utils import train_loop, setup_training

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a ControlNet model")
    
    # Model type selection
    parser.add_argument(
        "--model_type",
        type=str,
        default="controlnet",
        choices=["controlnet", "flux"],
        help="Type of ControlNet model to train: 'controlnet' for standard ControlNet or 'flux' for Flux ControlNet.",
    )
    
    # Model arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    
    # Training arguments
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
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for gradient clipping.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="constant",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using the `--resume_from_checkpoint` argument."
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the"
            " main process."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--control_dir",
        type=str,
        default=None,
        help="Directory containing control images.",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default=None,
        help="Directory containing target images.",
    )
    parser.add_argument(
        "--caption_file",
        type=str,
        default=None,
        help="JSON file containing image captions.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the HuggingFace dataset to use.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use.",
    )
    
    # Additional arguments for Flux ControlNet
    parser.add_argument(
        "--num_double_layers",
        type=int,
        default=4,
        help="Number of double layers in the Flux ControlNet (default: 4).",
    )
    parser.add_argument(
        "--num_single_layers",
        type=int,
        default=4,
        help="Number of single layers in the Flux ControlNet (default: 4).",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="The guidance scale used for Flux ControlNet.",
    )
    
    # Additional arguments for standard ControlNet
    parser.add_argument(
        "--num_extra_conditioning_channels",
        type=int,
        default=0,
        help="Number of extra conditioning channels for standard ControlNet.",
    )
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
    
    # Import the appropriate training module based on model_type
    if args.model_type == "controlnet":
        logger.info("Using standard ControlNet training")
        from relight.training import train_controlnet_sd3
        train_controlnet_sd3.main(args)
    elif args.model_type == "flux":
        logger.info("Using Flux ControlNet training")
        from relight.training import train_controlnet_flux
        train_controlnet_flux.main(args)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

if __name__ == "__main__":
    main() 