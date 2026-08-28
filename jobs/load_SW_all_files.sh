#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=Li9_hits_multi
#SBATCH --output=/scratch/elena/9Li/results/log/task_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/task_%A_%a.err
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

CHUNK_SIZE=25000
SCRIPT=/scratch/elena/9Li/scripts/load_and_sliding_windows.py
TASK_ID=${SLURM_ARRAY_TASK_ID}
BASE_ROOT="/scratch/elena/9Li/filtered_root"
CHUNK_MAP_DIR="/scratch/elena/9Li/chunk_maps"

# ---------------------------------------------------------------
# Signal / Background configuration
# ---------------------------------------------------------------

if [[ "$EXTRA_ARGS" == "--bkg" ]]; then
    CHUNK_MAP="$CHUNK_MAP_DIR/bkg_chunks.pkl"
    SAMPLE_NAME="BACKGROUND"
else
    CHUNK_MAP="$CHUNK_MAP_DIR/signal_chunks.pkl"
    SAMPLE_NAME="SIGNAL"
fi

echo "================================================================"
echo "Li9 sliding-window processing"
echo "================================================================"
echo "Sample      : $SAMPLE_NAME"
echo "Task ID     : $TASK_ID"
echo "Chunk map   : $CHUNK_MAP"
echo "Chunk size  : $CHUNK_SIZE"
echo "================================================================"

# ---------------------------------------------------------------
# Check that the chunk map exists
# ---------------------------------------------------------------

if [ ! -f "$CHUNK_MAP" ]; then
    echo "ERROR: Chunk map does not exist:"
    echo "$CHUNK_MAP"
    exit 1
fi

# ---------------------------------------------------------------
# Retrieve exact chunk information from the spill-aware map
# ---------------------------------------------------------------

read -r TARGET_RUN TARGET_PATH TARGET_CHUNK ENTRY_START ENTRY_STOP < <(
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
# Check for out-of-range task
# ---------------------------------------------------------------

if [ "$TARGET_RUN" == "EOF" ] || [ -z "$TARGET_RUN" ]; then
    echo "Task ID ${TASK_ID} exceeds total chunks. Exiting cleanly."
    exit 0
fi

# ---------------------------------------------------------------
# Reconstruct ROOT directory from momentum directory
# ---------------------------------------------------------------

TARGET_PATH="$BASE_ROOT/$TARGET_PATH"

# ---------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------

OUTDIR=/scratch/elena/9Li/results/run${TARGET_RUN}/processed
mkdir -p "$OUTDIR"

# ---------------------------------------------------------------
# Print task information
# ---------------------------------------------------------------

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')]"
echo "Global Task       = ${TASK_ID}"
echo "Sample            = ${SAMPLE_NAME}"
echo "Run               = ${TARGET_RUN}"
echo "Momentum path     = ${TARGET_PATH}"
echo "Chunk ID          = ${TARGET_CHUNK}"
echo "Entry start       = ${ENTRY_START}"
echo "Entry stop        = ${ENTRY_STOP}"
echo "Entries           = $((ENTRY_STOP - ENTRY_START))"
echo "Output directory  = ${OUTDIR}"
echo ""

# ---------------------------------------------------------------
# Check input directory
# ---------------------------------------------------------------

if [ ! -d "$TARGET_PATH" ]; then
    echo "ERROR: Target path does not exist:"
    echo "$TARGET_PATH"
    exit 1
fi

# ---------------------------------------------------------------
# Execute sliding-window analysis
# ---------------------------------------------------------------

python3 "$SCRIPT" \
    --run "$TARGET_RUN" \
    --chunk-id "$TARGET_CHUNK" \
    --chunk-size "$CHUNK_SIZE" \
    --entry-start "$ENTRY_START" \
    --entry-stop "$ENTRY_STOP" \
    --outdir "$OUTDIR" \
    --base-path "$TARGET_PATH" \
    $EXTRA_ARGS \
    --verbose

STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo ""
    echo "ERROR: Sliding-window processing failed."
    echo "Run=${TARGET_RUN}, Chunk=${TARGET_CHUNK}"
    exit $STATUS
fi

echo ""
echo "Task finished successfully:"
echo "  Sample = ${SAMPLE_NAME}"
echo "  Run    = ${TARGET_RUN}"
echo "  Chunk  = ${TARGET_CHUNK}"
echo "================================================================"