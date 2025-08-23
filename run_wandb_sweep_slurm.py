from simple_slurm import Slurm
import os

# ------------------- USER CONFIGURATION -------------------
SWEEP_ID = "orkozlovsky-tel-aviv-university/train-controlnet/d1ybu3j4"  # <-- Replace with your actual sweep id from `wandb sweep ...`
NUM_AGENTS = 10                   # Number of parallel agents/jobs to launch
CONDA_ENV = "relight_blender"    # Your conda environment name
PROJECT_DIR = "/home/dcor/orkozlovsky/repos/relight"  # Your project directory
# ----------------------------------------------------------

slurm = Slurm(
    job_name='wandb_agent',
    output='slurm/%j_wandb_agent.out',
    error='slurm/%j_wandb_agent.err',
    time='24:00:00',
    gres='gpu:1',
    mem='48G',
    partition='killable',
    account='gpu-research',
    chdir=PROJECT_DIR,
    constraint='h100|a6000|l40s|quadro_rtx_8000'
)

os.makedirs('slurm', exist_ok=True)

# Activate the conda environment
os.system(f'eval "$(conda shell.bash hook)" && conda activate {CONDA_ENV}')

# Set HuggingFace cache directory to a user-writable location
os.environ['HF_HOME'] = '/home/dcor/orkozlovsky/.cache/huggingface'

# Add relight source directory to PYTHONPATH
os.environ['PYTHONPATH'] = PROJECT_DIR + ':' + os.environ.get('PYTHONPATH', '')

# Command to run the wandb agent
wandb_agent_cmd = f'wandb agent {SWEEP_ID}'

for i in range(NUM_AGENTS):
    job_id = slurm.sbatch(wandb_agent_cmd)
    print(f"Submitted SLURM job {job_id} for wandb agent {i+1}/{NUM_AGENTS}")
    print(f"SLURM output will be in: slurm/{job_id}_wandb_agent.out")
    print("command: ", wandb_agent_cmd) 