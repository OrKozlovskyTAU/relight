from simple_slurm import Slurm
import os

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
    nodes=1,
    ntasks_per_node=1,
    cpus_per_task=8,
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

# Model settings for easy configuration
MODEL_SETTINGS = {
    "sd15": {
        "model_type": "sd15",
        "pretrained_model_name_or_path": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "resolution": 512,
        "train_batch_size": 4,
        "noise_zero_frequency_factor": 0.4,
        "learning_rate": 1e-4,
        "rescale_betas_zero_snr": False,
        "timestep_spacing": "leading",
        "guidance_rescale": 0.0,
    },
    "sd21": {
        "model_type": "sd21",
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-2-1",
        "resolution": 768,
        "train_batch_size": 4,
        "noise_zero_frequency_factor": 0,
        "learning_rate": 1e-4,
        "rescale_betas_zero_snr": True,
        "timestep_spacing": "trailing",
        "guidance_rescale": 0.7,
    },
}

# Select which model to use: "sd15" or "sd21"
SELECTED_MODEL = "sd21"
model_cfg = MODEL_SETTINGS[SELECTED_MODEL]

# Unique output directory and slurm output/error files per job
job_suffix = f"{model_cfg['model_type']}_noise{model_cfg['noise_zero_frequency_factor']}_lr{model_cfg['learning_rate']}"
output_dir = f"models/controlnet_{job_suffix}"
slurm_output = f"slurm/%j_{job_suffix}.out"
slurm_error = f"slurm/%j_{job_suffix}.err"

# Update slurm object for this job
slurm.output = slurm_output
slurm.error = slurm_error

# Build the command for this job
train_script = f'accelerate launch \
--config_file accelerate_config.yaml \
--main_process_port 14100 \
relight/training/train_controlnet.py \
--model_type="{model_cfg["model_type"]}" \
--pretrained_model_name_or_path="{model_cfg["pretrained_model_name_or_path"]}" \
--output_dir="{output_dir}" \
--train_data_dir="data_v3/train" \
--validation_data_dir="data_v3/val" \
--subset_size=5438 \
--max_validation_samples=20 \
--resolution={model_cfg["resolution"]} \
--learning_rate={model_cfg["learning_rate"]} \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--train_batch_size={model_cfg["train_batch_size"]} \
--max_train_steps=20000 \
--validation_steps=1000 \
--log_grad_and_weights_steps=1000 \
--num_validation_images=3 \
--validation_num_inference_steps=50 \
--noise_zero_frequency_factor={model_cfg["noise_zero_frequency_factor"]} \
--rescale_betas_zero_snr={model_cfg["rescale_betas_zero_snr"]} \
--timestep_spacing={model_cfg["timestep_spacing"]} \
--guidance_rescale={model_cfg["guidance_rescale"]}'

# Submit the job
job_id = slurm.sbatch(train_script)
print(f"Submitted SLURM job {job_id} for noise_zero_frequency_factor={model_cfg['noise_zero_frequency_factor']}")
print(f"SLURM output will be in: slurm/{job_id}_{job_suffix}.out")
print("command: ", train_script)
