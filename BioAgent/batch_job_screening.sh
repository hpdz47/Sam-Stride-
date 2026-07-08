#!/bin/bash
#SBATCH --gres=gpu:h200_nvl:1
#SBATCH -p cuda
#SBATCH -t 00-04:00:00
#SBATCH --job-name=hg_screen
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hlgp63@durham.ac.uk

# Host-guest SCREENING run (M2/M3): runs only the decision phase + interpreter,
# not the full plan/code/report pipeline. Mirrors batch_job.sh but launches
# run_screening.py. Requires the ground-truth bundle in $PROJECT/Inputs so that
# /inputs/DATA/... and /inputs/RAW-NMR/... exist.

echo "=== GPU check ==="
nvidia-smi || echo "nvidia-smi failed"
echo "================="

set -e

# Load secrets (.env must define LLM_Model_Reasoning, API_KEY, DOCKER_* )
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

CONTAINER_DIR=/nobackup/$USER/containers
CONTAINER=$CONTAINER_DIR/bioagent.sif
DOCKER_IMAGE=docker://sts2102/meng_project:latest

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
export SINGULARITY_DOCKER_USERNAME=$DOCKER_USERNAME
export SINGULARITY_DOCKER_PASSWORD=$DOCKER_TOKEN

mkdir -p "$CONTAINER_DIR" "$CACHE" "$TMPDIR" "$SINGULARITY_CACHEDIR" "$VLLM_CACHE" "$TRITON_CACHE" "$OUTPUTS"

echo "=== Checking container ==="
if [ ! -f "$CONTAINER" ]; then
    echo "Container not found. Pulling from DockerHub..."
    singularity pull "$CONTAINER" "$DOCKER_IMAGE"
else
    echo "Container already exists. Skipping pull."
fi

echo "=== Running Host-Guest Screening ==="
# SCREEN_WORKFLOW: SUPRAMOL-SCREENING (default) | SUPRAMOL-REPLICATION | SUPRAMOL-HOST-GUEST
# SCREEN_USE_PROCESSED: 1 = TopSpin 1r (exact), 0 = portable raw-FID path
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
  --env SCREEN_WORKFLOW="${SCREEN_WORKFLOW:-SUPRAMOL-SCREENING}" \
  --env SCREEN_USE_PROCESSED="${SCREEN_USE_PROCESSED:-1}" \
  "$CONTAINER" \
  python3 /app/run_screening.py
