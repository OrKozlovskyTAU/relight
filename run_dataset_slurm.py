from simple_slurm import Slurm
import os

# SLURM parameters
slurm = Slurm(
    job_name='relight_light_dataset',
    output='slurm/%j.out',
    error='slurm/%j.err',
    time='24:00:00',
    gres='gpu:1',
    mem='24G',
    partition='killable',
    account='gpu-research',
    chdir='/home/dcor/orkozlovsky/repos/relight',
    exclude='rack-omerl-g01,n-302,n-301',
    constraint='h100|a6000|l40s|quadro_rtx_8000|a5000'
)

# Ensure slurm output directory exists
os.makedirs('slurm', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Activate the conda environment
os.system('eval "$(conda shell.bash hook)" && conda activate relight_blender')

# Set HuggingFace cache directory to a user-writable location
os.environ['HF_HOME'] = '/home/dcor/orkozlovsky/.cache/huggingface'

# Add relight source directory to PYTHONPATH
os.environ['PYTHONPATH'] = '/home/dcor/orkozlovsky/repos/relight/:' + os.environ.get('PYTHONPATH', '')

# Build the command to run
cmd = '/home/dcor/orkozlovsky/blender-3.6.5-linux-x64/blender mpi_cornel.blend --background --python relight/cli/light_dataset.py -- --N 16 --Y 10 --output-dir data_v3'

# Run the command in terminal
# print(f"Running command: {cmd}")
# os.system(cmd)

# Submit the job
job_id = slurm.sbatch(cmd)
print(f"Submitted SLURM job {job_id}")
print(f"SLURM output will be in: slurm/{job_id}.out") 