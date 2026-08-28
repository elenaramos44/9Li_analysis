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

echo "================================================================"
echo "WCSim environment setup ready (Gd Stage 1)"
echo "Time: $(date)"
echo "================================================================"

CHUNK_SIZE=25000
SCRIPT=/scratch/elena/9Li/scripts/load_and_sliding_windows.py
TASK_ID=${SLURM_ARRAY_TASK_ID}

# ------------------------------------------------------------------------------
# Global spill-aware chunk map
# ------------------------------------------------------------------------------

CHUNK_MAP="${CHUNK_MAP}"

if [ -z "$CHUNK_MAP" ] || [ ! -f "$CHUNK_MAP" ]; then
    echo "ERROR: CHUNK_MAP not found:"
    echo "  $CHUNK_MAP"
    exit 1
fi

echo "Using chunk map:"
echo "  $CHUNK_MAP"
echo "SLURM array task:"
echo "  $TASK_ID"

# ------------------------------------------------------------------------------
# Read chunk information from the global chunk map
# ------------------------------------------------------------------------------

read -r TARGET_RUN TARGET_PATH TARGET_CHUNK ENTRY_START ENTRY_STOP < <(
python3 -c "
import pickle
import sys

chunk_map = '$CHUNK_MAP'
task_id = $TASK_ID

with open(chunk_map, 'rb') as f:
    chunks = pickle.load(f)

if task_id >= len(chunks):
    print('EOF EOF EOF EOF EOF')
    sys.exit(0)

chunk = chunks[task_id]

print(
    chunk['run'],
    chunk['file_path'],
    chunk['chunk_id'],
    chunk['entry_start'],
    chunk['entry_stop']
)
"
)

# ------------------------------------------------------------------------------
# Check whether this task corresponds to a valid chunk
# ------------------------------------------------------------------------------

if [ "$TARGET_RUN" == "EOF" ] || [ -z "$TARGET_RUN" ]; then
    echo "Task ID ${TASK_ID} exceeds total chunks for Gd."
    echo "Exiting cleanly."
    exit 0
fi

echo "----------------------------------------------------------------"
echo "Chunk information"
echo "----------------------------------------------------------------"
echo "Global task ID : ${TASK_ID}"
echo "Run            : ${TARGET_RUN}"
echo "ROOT path      : ${TARGET_PATH}"
echo "Chunk ID       : ${TARGET_CHUNK}"
echo "Entry start    : ${ENTRY_START}"
echo "Entry stop     : ${ENTRY_STOP}"
echo "Entries        : $((ENTRY_STOP - ENTRY_START))"
echo "----------------------------------------------------------------"

# ------------------------------------------------------------------------------
# Output directory
# ------------------------------------------------------------------------------

OUTDIR=/scratch/elena/9Li/results/run${TARGET_RUN}/processed

mkdir -p "$OUTDIR"

# ------------------------------------------------------------------------------
# Background / signal configuration
# ------------------------------------------------------------------------------

if [ "$EXTRA_ARGS" == "--bkg" ]; then
    echo "Sample type: GADOLINIUM BACKGROUND"
else
    echo "Sample type: GADOLINIUM SIGNAL"
fi

# ------------------------------------------------------------------------------
# Run Stage 1 Python script
# ------------------------------------------------------------------------------

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting load_and_sliding_windows.py"

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

# ------------------------------------------------------------------------------
# Check Python exit status
# ------------------------------------------------------------------------------

if [ $STATUS -ne 0 ]; then

    echo "================================================================"
    echo "ERROR: Stage 1 failed"
    echo "================================================================"
    echo "Global task ID : ${TASK_ID}"
    echo "Run            : ${TARGET_RUN}"
    echo "Chunk ID       : ${TARGET_CHUNK}"
    echo "Entry range    : ${ENTRY_START} - ${ENTRY_STOP}"
    echo "Exit status    : ${STATUS}"
    echo "================================================================"

    exit $STATUS
fi

echo "================================================================"
echo "Stage 1 task completed successfully"
echo "================================================================"
echo "Global task ID : ${TASK_ID}"
echo "Run            : ${TARGET_RUN}"
echo "Chunk ID       : ${TARGET_CHUNK}"
echo "Entry range    : ${ENTRY_START} - ${ENTRY_STOP}"
echo "Time           : $(date)"
echo "================================================================"

exit 0