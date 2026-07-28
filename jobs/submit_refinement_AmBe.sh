#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=AmBe_refine_parallel
#SBATCH --output=/scratch/elena/9Li/results/log/refine_task_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/refine_task_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4         
#SBATCH --mem=8G                 
#SBATCH --time=02:00:00           

source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh
source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh
export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

echo "WCSim environment setup ready"

SCRIPT=/scratch/elena/9Li/scripts/refinement_all.py
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Configuración fija para el run de AmBe
TARGET_RUN=2384
TARGET_CHUNK=${TASK_ID}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Global Task=${TASK_ID} -> Starting Refinement for AmBe Run=${TARGET_RUN} Chunk=${TARGET_CHUNK}"

# Pasamos --run, --chunk-id y --bkg de forma directa
python3 $SCRIPT \
    --run $TARGET_RUN \
    --chunk-id $TARGET_CHUNK \
    --bkg

echo "Task finished successfully: run=${TARGET_RUN} chunk=${TARGET_CHUNK}"