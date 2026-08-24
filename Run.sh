#!/bin/bash

# ---------------- SLURM Job Settings ----------------

#SBATCH --job-name=RNA_train                # Job name for identification
#SBATCH --partition=all_gpu                 # Partition (queue) to submit to: 'k40m', 'a100' or 'a30'
#SBATCH --gres=gpu:a100:1                   # Request 1 a100 from the node

#SBATCH -t 4-0:00                           # Max wall time: 4 days
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log


#SBATCH --mem-per-gpu=64G                  # RAM 64GB per a100.
#SBATCH --cpus-per-gpu=12                  # 12 Cores per a100

# ---------------- Environment Setup ----------------

# Load the appropriate module
module load conda/24.11.1
conda activate env_cuda
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# ---------------- Job Execution --------------------

# Run the simulation using MPI with 4 processes
python -u main_Train_subModel.py --config exp/Etapa_3_DM_SA_DN_javier_z.json
