from simple_slurm import Slurm
import os

# SLURM parameters
slurm = Slurm(
    job_name='relight_controlnet',
    output='slurm/%j.out',
    error='slurm/%j.err',
    time='10:00:00',
    gres='gpu:1',
    mem='48G',
    partition='killable',
    account='gpu-research',
    chdir='/home/dcor/orkozlovsky/repos/relight',
    exclude='rack-omerl-g01,n-302,n-301',
    constraint='h100|a6000|l40s|quadro_rtx_8000|a5000'
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

# Build the command to run
train_script = 'accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=no \
  --dynamo_backend=no \
  --main_process_port=29500 \
  relight/training/train_controlnet.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --output_dir="models/controlnet" \
  --train_data_dir="data/train" \
  --validation_data_dir="data/train" \
  --max_validation_samples=5 \
  --resolution=512 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --train_batch_size=4 \
  --max_train_steps=100 \
  --validation_steps=10 \
  --num_validation_images=3 \
  --validation_num_inference_steps=50 \
  --mse_loss_weight=0.0 \
  --mae_loss_weight=1.0 \
  --perceptual_loss_weight=0.0 \
  --log_training_image_steps=1000 \
  --log_grad_and_weights_steps=10'
    
# Submit the job
job_id = slurm.sbatch(train_script)
print(f"Submitted SLURM job {job_id}")
print(f"SLURM output will be in: slurm/{job_id}.out")
