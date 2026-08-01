#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=Li9_refine_Gd
#SBATCH --output=/scratch/elena/9Li/results/log/refine_task_Gd_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/refine_task_Gd_%A_%a.err
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

echo "WCSim environment setup ready (Refinement Gd)"

EXTRA_ARGS=${1:-$EXTRA_ARGS}
SCRIPT=/scratch/elena/9Li/scripts/refinement_all.py
TASK_ID=${SLURM_ARRAY_TASK_ID}
RESULTS_DIR="/scratch/elena/9Li/results"

# Runs de Gadolinio (6 runs)
RUNS=(2407 2408 2409 2432 2434 2438)

# ==============================================================================
# CÁLCULO DINÁMICO DE CHUNKS EXISTENTES EN DISCO POR RUN
# ==============================================================================
CHUNKS_PER_RUN=()

if [[ "$EXTRA_ARGS" == *"--bkg"* ]]; then
    echo ">> REFINEMENT (Gd): BACKGROUND MODE DETECTED <<"
    for RUN in "${RUNS[@]}"; do
        NUM=$(ls ${RESULTS_DIR}/run${RUN}/processed/Li9_clusters_chunk_*_BKG_multilat.pkl 2>/dev/null | wc -l)
        CHUNKS_PER_RUN+=($NUM)
    done
else
    echo ">> REFINEMENT (Gd): SIGNAL MODE DETECTED <<"
    for RUN in "${RUNS[@]}"; do
        NUM=$(ls ${RESULTS_DIR}/run${RUN}/processed/Li9_clusters_chunk_*_multilat.pkl 2>/dev/null | grep -v BKG | wc -l)
        CHUNKS_PER_RUN+=($NUM)
    done
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
    echo "Task ID ${TASK_ID} exceeds required chunks for Gd refinement. Exiting cleanly."
    exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Global Task Gd=${TASK_ID} -> Starting Refinement for Run=${TARGET_RUN} Chunk=${TARGET_CHUNK}"

# Ejecución del script Python
python3 $SCRIPT \
    --run $TARGET_RUN \
    --chunk-id $TARGET_CHUNK \
    $EXTRA_ARGS

echo "Task finished successfully: Gd run=${TARGET_RUN} chunk=${TARGET_CHUNK}"