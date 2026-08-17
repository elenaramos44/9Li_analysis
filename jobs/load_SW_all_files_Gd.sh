#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=Li9_hits_Gd
#SBATCH --output=/scratch/elena/9Li/results/log/task_Gd_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/task_Gd_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00

source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh
source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh
export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

echo "WCSim environment setup ready (Gd Stage 1)"

CHUNK_SIZE=25000
SCRIPT=/scratch/elena/9Li/scripts/load_and_sliding_windows.py
TASK_ID=${SLURM_ARRAY_TASK_ID}

#RUNS=(2407 2408 2409 2432 2434 2438)
#GD_BASE_PATH="/scratch/elena/9Li/filtered_root/Gd/p_270"

RUNS=(2374 2379)
GD_BASE_PATH="/scratch/elena/9Li/filtered_root/Gd/p_350"

# ------------------------------------------------------------------------------
# Determinar sufijo y flag para Python según la muestra
# ------------------------------------------------------------------------------
SAMPLE_SUFFIX="signal"
SAMPLE_FLAG=""

if [[ "$EXTRA_ARGS" == *"--bkg"* ]]; then
    SAMPLE_SUFFIX="bkg"
    SAMPLE_FLAG="--bkg"
    echo ">> SLIDING WINDOWS (Gd): BACKGROUND MODE DETECTED <<"
else
    echo ">> SLIDING WINDOWS (Gd): SIGNAL MODE DETECTED <<"
fi

# ------------------------------------------------------------------------------
# Cálculo DINÁMICO del número de chunks exactos por cada RUN de Gd
# ------------------------------------------------------------------------------
CHUNKS_PER_RUN=()
for RUN in "${RUNS[@]}"; do
    FILE="${GD_BASE_PATH}/WCTE_merged_production_R${RUN}_${SAMPLE_SUFFIX}.root"
    NUM_CHUNKS=$(python3 -c "
import uproot, math, os
if os.path.exists('${FILE}'):
    with uproot.open('${FILE}') as f:
        entries = f['WCTEReadoutWindows'].num_entries
        print(math.ceil(entries / ${CHUNK_SIZE}))
else:
    print(0)
")
    CHUNKS_PER_RUN+=($NUM_CHUNKS)
done

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
    echo "Task ID ${TASK_ID} exceeds total required chunks for Gd. Exiting cleanly."
    exit 0
fi

OUTDIR=/scratch/elena/9Li/results/run${TARGET_RUN}/processed
mkdir -p $OUTDIR

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Global Task=${TASK_ID} -> Processing Gd Run=${TARGET_RUN} Chunk=${TARGET_CHUNK}"

# Ejecución de Python usando únicamente $SAMPLE_FLAG (evitando --is-gd)
python3 $SCRIPT \
    --run $TARGET_RUN \
    --chunk-id $TARGET_CHUNK \
    --chunk-size $CHUNK_SIZE \
    --outdir $OUTDIR \
    --base-path $GD_BASE_PATH \
    $SAMPLE_FLAG \
    --verbose

echo "Task finished successfully: Gd run=${TARGET_RUN} chunk=${TARGET_CHUNK}"