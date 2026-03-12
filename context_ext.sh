#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=65536
#SBATCH --time=168:00:00
#SBATCH --gpus=A100:4
#SBATCH -o logs/%j.log
#SBATCH -e logs/%j.err
module load dev2025a multigpu cuda/12.6.3
source ~/hpc_notify.sh

export HF_HOME="/scratch/sraj/"
export TRITON_CACHE_DIR="/scratch/sraj/tritoncache"
# or
export TRITON_HOME="/scratch/sraj/tritoncache"
export VLLM_CACHE_ROOT="/scratch/sraj/vllmcache"

source .venv/bin/activate
hpc_notify "🚀 Started: MBert Training"
# python quick_gpu_debug.py
WANDB__SERVICE_WAIT=300 composer main.py yamls/modernbert/modernbert-base-context-extension.yaml 
# WANDB__SERVICE_WAIT=300 composer main.py yamls/modernbert/modernbert-base-context-extension_2k.yaml 

# TORCHDYNAMO_DISABLE=1
hpc_notify "✅ Completed: MBert Training"