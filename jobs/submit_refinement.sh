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


# ---------------------------------------------------------------
# Refinement script
# ---------------------------------------------------------------

SCRIPT=/scratch/elena/9Li/scripts/refinement_all.py

TASK_ID=${SLURM_ARRAY_TASK_ID}

BASE_DIR="/scratch/elena/9Li"
CHUNK_MAP_DIR="$BASE_DIR/chunk_maps"


# ---------------------------------------------------------------
# Select correct chunk map
# ---------------------------------------------------------------

EXTRA_ARGS=${1:-$EXTRA_ARGS}

if [[ "$EXTRA_ARGS" == "--bkg" ]]; then

    SAMPLE_NAME="BACKGROUND"
    CHUNK_MAP="$CHUNK_MAP_DIR/bkg_chunks.pkl"

else

    SAMPLE_NAME="SIGNAL"
    CHUNK_MAP="$CHUNK_MAP_DIR/signal_chunks.pkl"

fi


# ---------------------------------------------------------------
# Check chunk map
# ---------------------------------------------------------------

if [ ! -f "$CHUNK_MAP" ]; then

    echo "ERROR: Chunk map does not exist:"
    echo "$CHUNK_MAP"
    exit 1

fi


# ---------------------------------------------------------------
# Retrieve exact task from global spill-aware chunk map
# ---------------------------------------------------------------

read -r TARGET_RUN TARGET_MOMENTUM TARGET_CHUNK TARGET_START TARGET_STOP < <(
python3 - "$CHUNK_MAP" "$TASK_ID" <<'PY'
import sys
import pickle

chunk_map = sys.argv[1]
task_id = int(sys.argv[2])

with open(chunk_map, "rb") as f:
    chunks = pickle.load(f)

if task_id < 0 or task_id >= len(chunks):
    print("EOF EOF EOF EOF EOF")
    sys.exit(0)

chunk = chunks[task_id]

print(
    chunk["run"],
    chunk["momentum_dir"],
    chunk["chunk_id"],
    chunk["entry_start"],
    chunk["entry_stop"]
)
PY
)


# ---------------------------------------------------------------
# Check for invalid task
# ---------------------------------------------------------------

if [ "$TARGET_RUN" == "EOF" ] || [ -z "$TARGET_RUN" ]; then

    echo "Task ID ${TASK_ID} exceeds total chunks."
    echo "Exiting cleanly."

    exit 0

fi


# ---------------------------------------------------------------
# Print task information
# ---------------------------------------------------------------

echo "==============================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Stage 4 refinement"
echo "==============================================================="
echo "Global Task : $TASK_ID"
echo "Sample      : $SAMPLE_NAME"
echo "Run         : $TARGET_RUN"
echo "Momentum    : $TARGET_MOMENTUM"
echo "Chunk       : $TARGET_CHUNK"
echo "Entry start : $TARGET_START"
echo "Entry stop  : $TARGET_STOP"
echo "==============================================================="


# ---------------------------------------------------------------
# Run refinement
# ---------------------------------------------------------------

python3 "$SCRIPT" \
    --run "$TARGET_RUN" \
    --chunk-id "$TARGET_CHUNK" \
    $EXTRA_ARGS

STATUS=$?


# ---------------------------------------------------------------
# Check refinement status
# ---------------------------------------------------------------

if [ $STATUS -ne 0 ]; then

    echo ""
    echo "ERROR: Refinement failed."
    echo "Run=${TARGET_RUN}"
    echo "Momentum=${TARGET_MOMENTUM}"
    echo "Chunk=${TARGET_CHUNK}"

    exit $STATUS

fi


echo ""
echo "Task finished successfully:"
echo "  Sample   = ${SAMPLE_NAME}"
echo "  Run      = ${TARGET_RUN}"
echo "  Momentum = ${TARGET_MOMENTUM}"
echo "  Chunk    = ${TARGET_CHUNK}"
echo "==============================================================="