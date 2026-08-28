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


# ---------------------------------------------------------------
# Multilateration script
# ---------------------------------------------------------------

SCRIPT=/scratch/elena/9Li/scripts/multilat_vertex_reconstruction.py

TASK_ID=${SLURM_ARRAY_TASK_ID}

BASE_DIR="/scratch/elena/9Li"
CHUNK_MAP_DIR="$BASE_DIR/chunk_maps"


# ---------------------------------------------------------------
# Select correct chunk map
# ---------------------------------------------------------------

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
# Input/output directories
# ---------------------------------------------------------------

IN_DIR="/scratch/elena/9Li/results/run${TARGET_RUN}/processed"
OUT_DIR="/scratch/elena/9Li/results/run${TARGET_RUN}/processed"


# ---------------------------------------------------------------
# Select Stage 1 output PKL
# ---------------------------------------------------------------

if [[ "$EXTRA_ARGS" == "--bkg" ]]; then

    INPUT_FILE="${IN_DIR}/Li9_clusters_chunk_${TARGET_CHUNK}_BKG.pkl"

else

    INPUT_FILE="${IN_DIR}/Li9_clusters_chunk_${TARGET_CHUNK}.pkl"

fi


# ---------------------------------------------------------------
# Check that Stage 1 output exists
# ---------------------------------------------------------------

if [ ! -f "$INPUT_FILE" ]; then

    echo "ERROR: Expected Stage 1 PKL does not exist:"
    echo "$INPUT_FILE"
    echo ""
    echo "Run      : $TARGET_RUN"
    echo "Momentum : $TARGET_MOMENTUM"
    echo "Chunk    : $TARGET_CHUNK"
    exit 1

fi


mkdir -p "$OUT_DIR"


# ---------------------------------------------------------------
# Print task information
# ---------------------------------------------------------------

echo "--------------------------------------------------------"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Global Task: $TASK_ID"
echo "Sample: $SAMPLE_NAME"
echo "Run: $TARGET_RUN"
echo "Momentum: $TARGET_MOMENTUM"
echo "Global Chunk: $TARGET_CHUNK"
echo "Entry range: $TARGET_START - $TARGET_STOP"
echo "Input PKL: $INPUT_FILE"
echo "Output Dir: $OUT_DIR"
echo "--------------------------------------------------------"


# ---------------------------------------------------------------
# Run multilateration
# ---------------------------------------------------------------

python3 "$SCRIPT" \
    --pkl "$INPUT_FILE" \
    --outdir "$OUT_DIR" \
    $EXTRA_ARGS \
    --verbose

STATUS=$?

if [ $STATUS -ne 0 ]; then

    echo ""
    echo "ERROR: Multilateration failed."
    echo "Run=${TARGET_RUN}"
    echo "Momentum=${TARGET_MOMENTUM}"
    echo "Chunk=${TARGET_CHUNK}"
    exit $STATUS

fi


echo ""
echo "Finished global chunk ${TARGET_CHUNK}"
echo "Run ${TARGET_RUN}"
echo "Momentum ${TARGET_MOMENTUM}"
echo "========================================================"