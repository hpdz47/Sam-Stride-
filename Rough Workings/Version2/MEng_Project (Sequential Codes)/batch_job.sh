#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:ampere:1
#SBATCH -p ug-gpu-small
#SBATCH --qos=normal
#SBATCH -t 00-10:00:00
#SBATCH --job-name=hlgp63
#SBATCH --mail-type=[BEGIN,END,FAIL]
#SBATCH --mail-user hlgp63@durham.ac.uk
#SBATCH --mem=28G

module purge
# Load CUDA 11.8 with cuDNN 8.7 - specific version to match environment.yml
module load cuda/11.8-cudnn8.7


# Initialize conda
source /home3/hlgp63/anaconda3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate MENG_env

# Run vLLM Installation inside environment AND upgrade relevant packages.
pip install --upgrade uv -qq
uv pip install -U vllm --torch-backend=auto -qq

pip show vllm
pip show ag2
#pip install --upgrade tabpfn -qq
#pip install --upgrade tabpfn-extensions[all] -qq

# Stops export of browser usage stats.
export ANONYMIZED_TELEMETRY=false
#export TABPFN_DISABLE_TELEMETRY=1

# Add these diagnostic lines first
echo "=== Environment Information ==="
echo "Modules loaded:"
module list
echo "Python path: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

echo "=== GPU Information ==="
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA Version: $(nvcc --version 2>/dev/null || echo 'nvcc not found')"
echo ""

# Try different GPU detection methods
echo "Method 1: nvidia-smi"
nvidia-smi || echo "nvidia-smi failed"
echo ""

echo "Method 2: nvidia-debugdump"
nvidia-debugdump -l || echo "nvidia-debugdump failed"
echo "======================"

# Export Relevant Environment Variables:
source .env
export HF_TOKEN

# Run the script with increased CUDA memory settings

echo "Starting Python script..."
python Data_Analysis_Conversation.py



