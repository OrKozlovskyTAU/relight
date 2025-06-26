from time import sleep
from simple_slurm import Slurm
import os
from itertools import product

# -----------------------------------------------------------------

# Available GPU Constraints and VRAM (Descending Order):
# -----------------------------------------------------------------
# Constraint           | VRAM (GiB)
# -----------------------------------------------------------------
# h100                 | 80
# a6000                | 48
# l40s                 | 48
# quadro_rtx_8000      | 48 
# tesla_v100           | 32
# a5000                | 24
# geforce_rtx_3090     | 24
# titan_xp             | 12 
# geforce_rtx_2080     | 11 
# Example Usage in sbatch script with multiple options (request any of the listed GPUs):
# #SBATCH --constraint="h100|a6000|L40S"  # Request an H100, A6000, or L40S GPU (80, 48, or 48 GiB VRAM respectively)
# -----------------------------------------------------------------

# SLURM parameters
slurm = Slurm(
    job_name='relight_controlnet',
    output='slurm/%j.out',
    error='slurm/%j.err',
    time='24:00:00',
    gres='gpu:1',
    mem='48G',
    partition='killable',
    account='gpu-research',
    chdir='/home/dcor/orkozlovsky/repos/relight',
    constraint='h100|a6000|l40s|quadro_rtx_8000'
)

# Ensure slurm output directory exists
os.makedirs('slurm', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Activate the conda environment
os.system('eval "$(conda shell.bash hook)" && conda activate relight_blender')

# Set HuggingFace cache directory to a user-writable location
os.environ['HF_HOME'] = '/home/dcor/orkozlovsky/.cache/huggingface'

# Add relight source directory to PYTHONPATH
os.environ['PYTHONPATH'] = '/home/dcor/orkozlovsky/repos/relight/:' + os.environ.get('PYTHONPATH', '')

# Define combinations of loss weights to try
loss_weight_combinations = [
    # (mse_loss_weight, mae_loss_weight, perceptual_loss_weight)
    (1.0, 0.0, 0.0),
]

# Zero frequency factors to try
zero_frequency_factors = [
    0.4,
]

learning_rates = [
    5e-4,
]
            
i = 0
# Submit a SLURM job for each value of noise_zero_frequency_factor
for mse_loss_weight, mae_loss_weight, perceptual_loss_weight in loss_weight_combinations:
    for noise_zero_frequency_factor in zero_frequency_factors:
        for learning_rate in learning_rates:
        
            i += 1
            # Unique output directory and slurm output/error files per job
            job_suffix = f"mse{mse_loss_weight}_mae{mae_loss_weight}_perc{perceptual_loss_weight}_noise{noise_zero_frequency_factor}_lr{learning_rate}"
            output_dir = f"models/controlnet_{job_suffix}"
            slurm_output = f"slurm/%j_{job_suffix}.out"
            slurm_error = f"slurm/%j_{job_suffix}.err"

            # Update slurm object for this job
            slurm.output = slurm_output
            slurm.error = slurm_error

            # Build the command for this job
            train_script = f'accelerate launch \
            --num_processes=1 \
            --num_machines=1 \
            --mixed_precision=no \
            --dynamo_backend=no \
            --main_process_port={10000 + i * 10} \
            relight/training/train_controlnet.py \
            --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
            --output_dir="{output_dir}" \
            --train_data_dir="data_v2/train" \
            --validation_data_dir="data_v2/val" \
            --subset_size=5438 \
            --max_validation_samples=20 \
            --resolution=512 \
            --learning_rate={learning_rate} \
            --lr_scheduler="constant" \
            --lr_warmup_steps=0 \
            --train_batch_size=4 \
            --max_train_steps=10000 \
            --validation_steps=1000 \
            --log_training_image_steps=150000 \
            --log_grad_and_weights_steps=1000 \
            --num_validation_images=3 \
            --validation_num_inference_steps=50 \
            --mse_loss_weight={mse_loss_weight} \
            --mae_loss_weight={mae_loss_weight} \
            --perceptual_loss_weight={perceptual_loss_weight} \
            --noise_zero_frequency_factor={noise_zero_frequency_factor} \
            --noise_scheduler_prediction_type="epsilon" \
            --lab_color_match_logging=True'

            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Submit the job
            job_id = slurm.sbatch(train_script)
            print(f"Submitted SLURM job {job_id} for loss weights: mse={mse_loss_weight}, mae={mae_loss_weight}, perc={perceptual_loss_weight}, noise_zero_frequency_factor={noise_zero_frequency_factor}")
            print(f"SLURM output will be in: slurm/{job_id}_{job_suffix}.out")
            print("command: ", train_script)
            # sleep(120)
