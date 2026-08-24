#!/bin/bash

# =========================================================
# GLOBAL RUN SETTINGS
# Altere aqui para atualizar todos os jobs da corrente
# =========================================================
export LBPM_VERSION="lbpm/cpu/lbpm_fork_2016010"
PARTITION="all_gpu"
GRES_STR="gpu:k40m:1"

# Configuração dinâmica de GRES
GRES_FLAG=""
if [ ! -z "$GRES_STR" ]; then
    GRES_FLAG="--gres=$GRES_STR"
fi

j1=$(sbatch --parsable --partition=$PARTITION $GRES_FLAG run_sims_00000_00000.sh)
echo "Submitted run_sims_00000_00000.sh to $PARTITION (Job: $j1)"

echo "--> All chained jobs submitted."
