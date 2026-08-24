#!/bin/bash

# ---------------- SLURM Job Settings ----------------

#SBATCH --job-name=RNA_train                # Job name for identification
#SBATCH --partition=close_cpu                 # Partition (queue) to submit to: 'k40m', 'a100' or 'a30'
#SBATCH --nodes=1                           # Number of nodes

#SBATCH -t 4-0:00                           # Max wall time: 4 days
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log



# ---------------- Environment Setup ----------------

# Load the appropriate module (with CUDA-aware MPI already built)
module load conda/24.11.1
conda activate env_cuda
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH # Makes the paths from conda environment visible
# ---------------- Job Execution --------------------

python main_TestModel_Quant.py
