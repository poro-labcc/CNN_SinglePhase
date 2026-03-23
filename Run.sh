#!/bin/bash

# ---------------- SLURM Job Settings ----------------

#SBATCH --job-name=RNA_train                # Job name for identification
#SBATCH --partition=all_gpu                 # Partition (queue) to submit to: 'k40m', 'a100' or 'a30'
##SBATCH --nodes=1                           # Number of nodes
#SBATCH --gres=gpu:a100:1                   # Request 1 a100 from the node

#SBATCH -t 4-0:00                           # Max wall time: 4 days
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log


#SBATCH --mem-per-gpu=64G                  # RAM 64GB per a100.
#SBATCH --cpus-per-gpu=12                  # 12 Cores per a100

##SBATCH --oversubscribe  	               # Allow sharing

# ---------------- Environment Setup ----------------

# Load the appropriate module (with CUDA-aware MPI already built)
module load conda/24.11.1
conda activate env_cuda
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH # Makes the paths from conda environment visible
# ---------------- Job Execution --------------------

# Run the simulation using MPI with 4 processes
python -u main_TrainModel_L1cos.py
