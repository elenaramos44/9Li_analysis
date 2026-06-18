#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=Li9_multilat
#SBATCH --output=/scratch/elena/9Li/results/log/multilat_task_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/multilat_task_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00



echo "Setting environment for multilateration"

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

echo "Environment ready (multilateration)"


SCRIPT=/scratch/elena/9Li/scripts/multilat_vertex_reconstruction.py
TASK_ID=${SLURM_ARRAY_TASK_ID}

RUNS=(1928 1930 1932 1934 1935 1936 1937 1938 1939 1941 1846 1848)

# ==============================================================================
# SELECCIÓN DINÁMICA DE CHUNKS SEGÚN EXTRA_ARGS
# ==============================================================================
if [[ "$EXTRA_ARGS" == "--bkg" ]]; then
    echo ">> MULTILATERATION: BACKGROUND MODE DETECTED <<"
    CHUNKS_PER_RUN=(55 80 64 48 47 39 52 31 52 63 18 16)
else
    echo ">> MULTILATERATION: SIGNAL MODE DETECTED <<"
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

IN_DIR="/scratch/elena/9Li/results/run${TARGET_RUN}/processed"
OUT_DIR="/scratch/elena/9Li/results/run${TARGET_RUN}/processed" # Guardamos todo en processed para simplificar rutas

# Definimos el nombre del archivo de entrada según la muestra procesada
if [[ "$EXTRA_ARGS" == "--bkg" ]]; then
    INPUT_FILE="${IN_DIR}/Li9_clusters_chunk_${TARGET_CHUNK}_BKG.pkl"
else
    INPUT_FILE="${IN_DIR}/Li9_clusters_chunk_${TARGET_CHUNK}.pkl"
fi

mkdir -p $OUT_DIR

echo "--------------------------------------------------------"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Global Task: $TASK_ID"
echo "Processing Run: $TARGET_RUN"
echo "Processing Chunk: $TARGET_CHUNK"
echo "Input PKL: $INPUT_FILE"
echo "Output Dir: $OUT_DIR"
echo "--------------------------------------------------------"

# Añadimos $EXTRA_ARGS al script de Python
python3 $SCRIPT \
    --pkl $INPUT_FILE \
    --outdir $OUT_DIR \
    $EXTRA_ARGS \
    --verbose

echo "Finished chunk ${TARGET_CHUNK} for run ${TARGET_RUN}"