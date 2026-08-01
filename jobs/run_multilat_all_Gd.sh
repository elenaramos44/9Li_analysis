#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=Li9_multilat_Gd
#SBATCH --output=/scratch/elena/9Li/results/log/multilat_task_Gd_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/multilat_task_Gd_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00

echo "Setting environment for multilateration (Gd)"

source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh
source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh

export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake
export WCSIM_BUILD_DIR=/scratch/elena/wcsim-install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

export BONSAIDIR=/scratch/elena/bonsai
export LD_LIBRARY_PATH=$BONSAIDIR:$LD_LIBRARY_PATH
export ROOT_INCLUDE_PATH=$BONSAIDIR/bonsai:/scratch/elena/wcsim-install/include/WCSim:$ROOT_INCLUDE_PATH

echo "Environment ready (multilateration Gd)"

SCRIPT=/scratch/elena/9Li/scripts/multilat_vertex_reconstruction.py
TASK_ID=${SLURM_ARRAY_TASK_ID}
RESULTS_DIR="/scratch/elena/9Li/results"

# Runs de Gadolinio (6 runs)
RUNS=(2407 2408 2409 2432 2434 2438)

# ==============================================================================
# CÁLCULO DINÁMICO DE CHUNKS EXISTENTES EN DISCO POR RUN
# ==============================================================================
CHUNKS_PER_RUN=()

if [[ "$EXTRA_ARGS" == *"--bkg"* ]]; then
    echo ">> MULTILATERATION (Gd): BACKGROUND MODE DETECTED <<"
    for RUN in "${RUNS[@]}"; do
        NUM=$(ls ${RESULTS_DIR}/run${RUN}/processed/Li9_clusters_chunk_*_BKG.pkl 2>/dev/null | wc -l)
        CHUNKS_PER_RUN+=($NUM)
    done
else
    echo ">> MULTILATERATION (Gd): SIGNAL MODE DETECTED <<"
    for RUNs in "${RUNS[@]}"; do
        # Pequeño apunte: aseguramos evaluar la variable correctamente
        :
    done
    for RUN in "${RUNS[@]}"; do
        NUM=$(ls ${RESULTS_DIR}/run${RUN}/processed/Li9_clusters_chunk_*.pkl 2>/dev/null | grep -v BKG | wc -l)
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
    echo "Task ID ${TASK_ID} exceeds required chunks for Gd. Exiting cleanly."
    exit 0
fi

IN_DIR="${RESULTS_DIR}/run${TARGET_RUN}/processed"
OUT_DIR="${RESULTS_DIR}/run${TARGET_RUN}/processed"

if [[ "$EXTRA_ARGS" == *"--bkg"* ]]; then
    INPUT_FILE="${IN_DIR}/Li9_clusters_chunk_${TARGET_CHUNK}_BKG.pkl"
else
    INPUT_FILE="${IN_DIR}/Li9_clusters_chunk_${TARGET_CHUNK}.pkl"
fi

mkdir -p $OUT_DIR

echo "--------------------------------------------------------"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Global Task Gd: $TASK_ID"
echo "Processing Run: $TARGET_RUN"
echo "Processing Chunk: $TARGET_CHUNK"
echo "Input PKL: $INPUT_FILE"
echo "Output Dir: $OUT_DIR"
echo "--------------------------------------------------------"

# Pasamos $EXTRA_ARGS directamente (que es '--bkg' o vacío)
python3 $SCRIPT \
    --pkl $INPUT_FILE \
    --outdir $OUT_DIR \
    $EXTRA_ARGS \
    --verbose

echo "Finished chunk ${TARGET_CHUNK} for run ${TARGET_RUN}"