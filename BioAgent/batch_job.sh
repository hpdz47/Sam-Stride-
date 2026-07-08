#!/bin/bash
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH -p cuda
#SBATCH -t 02-08:00:00
#SBATCH --job-name=hlgp63
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hlgp63@durham.ac.uk

# Try different GPU detection methods
echo "Method 1: nvidia-smi"
nvidia-smi || echo "nvidia-smi failed"
echo ""

echo "Method 2: nvidia-debugdump"
nvidia-debugdump -l || echo "nvidia-debugdump failed"
echo "======================"


set -e

# Load secrets
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

CONTAINER_DIR=/nobackup/$USER/containers
CONTAINER=$CONTAINER_DIR/bioagent.sif
DOCKER_IMAGE=docker://hpdz47/meng_project:latest

PROJECT=/nobackup/$USER/projects/BioAgent
CACHE=/nobackup/$USER/hf_cache
TMPDIR=/nobackup/$USER/tmp
OUTPUTS=$PROJECT/Outputs
VLLM_CACHE=/nobackup/$USER/vllm_cache     
TRITON_CACHE=/nobackup/$USER/triton_cache

export SINGULARITY_CACHEDIR=/nobackup/$USER/.singularity_cache
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/vllm/torchinductor
export TMPDIR
export SINGULARITY_TMPDIR=$TMPDIR

# DockerHub authentication
export SINGULARITY_DOCKER_USERNAME=$DOCKER_USERNAME
export SINGULARITY_DOCKER_PASSWORD=$DOCKER_TOKEN

mkdir -p "$CONTAINER_DIR"
mkdir -p "$CACHE"
mkdir -p "$TMPDIR"
mkdir -p "$SINGULARITY_CACHEDIR"
mkdir -p "$VLLM_CACHE"    
mkdir -p "$TRITON_CACHE"

echo "=== Checking container ==="

if [ ! -f "$CONTAINER" ]; then
    echo "Container not found. Pulling from DockerHub..."
    singularity pull "$CONTAINER" "$DOCKER_IMAGE"
else
    echo "Container already exists. Skipping pull."
fi

# Testing Installs Inside Container:
singularity exec $CONTAINER python3 -m pip show ag2

echo "=== Running BioAgent ==="

# Singularity Container Setup:
# 1. workspace - Name for directory that is passed to agents. Outputs such as Scratchpad, Repos, Logs, etc will be in this as read-write.
# 2. app - Main code files for system. Read-only to prevent LLM-damage.
# 3. inputs - Passed to agents as read only so they can access datasets to analyse but cannot edit.
# 4. cache directories that must be writable for model weights.

singularity exec --nv --cleanenv --containall \
  --bind "$OUTPUTS:/workspace" \
  --bind "$PROJECT:/app:ro" \
  --bind "$PROJECT/Inputs:/inputs:ro" \
  --bind "$CACHE:/root/.cache/huggingface" \
  --bind "$PROJECT/.env:/app/.env:ro" \
  --bind "$VLLM_CACHE:/root/.cache/vllm" \
  --bind "$TRITON_CACHE:/root/.triton" \
  --bind "$TMPDIR:/tmp" \
  --env VLLM_CACHE_ROOT=/root/.cache/vllm \
  --env TRITON_CACHE_DIR=/root/.triton \
  --env HF_HOME=/root/.cache/huggingface \
  --env TORCHINDUCTOR_CACHE_DIR=/root/.cache/vllm/torchinductor \
  "$CONTAINER" \
  python3 /app/BioAgent.py

