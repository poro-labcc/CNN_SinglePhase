#!/bin/bash

# ---------------- SLURM Job Settings ----------------
#SBATCH --oversubscribe
#SBATCH --job-name=Perm_00000_00000

#SBATCH -t 7-0:00
#SBATCH -o perm_00000_00000_%j.out
#SBATCH -e perm_00000_00000_%j.err
#SBATCH --ntasks=1

# ---------------- Environment Setup ----------------
module load $LBPM_VERSION

echo "=== Chunk 000 | Samples 00000 to 00000 ==="

echo "--- Launching simulation for Sample_00000 ---"
cd Sample_00000
echo "Current Simulation: " ${PWD##*/}
mpirun --oversubscribe -np 1 lbpm_permeability_simulator simulation.db

echo "--> All simulations in this chunk finished."
