#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=Li9_refine_parallel
#SBATCH --output=/scratch/elena/9Li/results/log/refine_task_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/refine_task_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4         
#SBATCH --mem=8G                 
#SBATCH --time=2:00:00           

source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh
source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh
export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

echo "WCSim environment setup ready"

# ==============================================================================
# CAPTURE EXTRA_ARGS FROM PIPELINE.SH
# ==============================================================================
EXTRA_ARGS=${1:-$EXTRA_ARGS}

SCRIPT=/scratch/elena/9Li/scripts/refinement_all.py
TASK_ID=${SLURM_ARRAY_TASK_ID}

RUNS=(1928 1930 1932 1934 1935 1936 1937 1938 1939 1941 1846 1848)

# ==============================================================================
# SELECCIÓN DINÁMICA DE CHUNKS SEGÚN EXTRA_ARGS
# ==============================================================================
if [[ "$EXTRA_ARGS" == "--bkg" ]]; then
    echo ">> REFINEMENT: BACKGROUND MODE DETECTED <<"
    CHUNKS_PER_RUN=(55 80 64 48 47 39 52 31 52 63 18 16)
else
    echo ">> REFINEMENT: SIGNAL MODE DETECTED <<"
    CHUNKS_PER_RUN=(24 13 34 19 22 20 27 13 21 31 33 33)
fi

CURRENT_SUM=0
TARGET_RUN=""
TARGET_CHUNK=""

for i in "${!RUNS[@]}"; do
    NUM_CHUNKS=${CHUNKS_PER_RUN[$i]}
    NEXT_SUM=$((CURRENT_SUM + NUM_CHUNKS))
    
    if [ "$TASK_ID" -lt "$NEXT_SUM" ]; then
        TARGET_RUN=${RUNS[$i]}
        TARGET_CHUNK=$((TASK_ID - CURRENT_SUM))
        break
    fi
    CURRENT_SUM=$NEXT_SUM
done

if [ -z "$TARGET_RUN" ]; then
    echo "Task ID ${TASK_ID} exceeds required chunks. Exiting cleanly."
    exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Global Task=${TASK_ID} -> Starting Refinement for Run=${TARGET_RUN} Chunk=${TARGET_CHUNK}"

# Añadimos $EXTRA_ARGS al final del comando de Python
python3 $SCRIPT \
    --run $TARGET_RUN \
    --chunk-id $TARGET_CHUNK \
    $EXTRA_ARGS

echo "Task finished successfully: run=${TARGET_RUN} chunk=${TARGET_CHUNK}"