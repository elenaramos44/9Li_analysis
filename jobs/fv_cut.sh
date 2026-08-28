#!/bin/bash

#SBATCH --qos=regular
#SBATCH --job-name=Li9_final_fv
#SBATCH --output=/scratch/elena/9Li/results/log/merge_task_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/merge_task_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=4:30:00


# ---------------------------------------------------------------
# Environment
# ---------------------------------------------------------------

source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh
source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh

export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

echo "WCSim environment setup ready"


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------

EXTRA_ARGS=${1:-$EXTRA_ARGS}

SCRIPT="/scratch/elena/9Li/scripts/merge_and_fv_cut.py"

INDEX=${SLURM_ARRAY_TASK_ID}

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

    echo ""
    echo "ERROR: Chunk map does not exist:"
    echo "$CHUNK_MAP"
    exit 1

fi


# ---------------------------------------------------------------
# Retrieve unique run corresponding to this array index
#
# Stage 5 operates at RUN level, unlike Stages 1, 2 and 4,
# which operate at CHUNK level.
#
# The run list is derived from the SAME spill-aware chunk map
# used by all previous stages.
# ---------------------------------------------------------------

TARGET_RUN=$(python3 - "$CHUNK_MAP" "$INDEX" <<'PY'
import sys
import pickle

chunk_map = sys.argv[1]
index = int(sys.argv[2])

with open(chunk_map, "rb") as f:
    chunks = pickle.load(f)

# Unique runs represented in the chunk map.
#
# Sorting gives the same deterministic run ordering used
# by submit_stage5.sh.
unique_runs = sorted(
    set(chunk["run"] for chunk in chunks)
)

if index < 0 or index >= len(unique_runs):
    print("NONE")
    sys.exit(0)

print(unique_runs[index])
PY
)


# ---------------------------------------------------------------
# Check for invalid array index
# ---------------------------------------------------------------

if [ "$TARGET_RUN" == "NONE" ] || [ -z "$TARGET_RUN" ]; then

    echo ""
    echo "ERROR: Array index out of bounds."
    echo "Stage 5 array index : ${INDEX}"
    echo "Chunk map           : ${CHUNK_MAP}"
    exit 1

fi


# ---------------------------------------------------------------
# Print task information
# ---------------------------------------------------------------

echo ""
echo "================================================================"
echo "Stage 5: Final Merge + Fiducial Volume Cut"
echo "================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]"
echo "Sample       : ${SAMPLE_NAME}"
echo "Array index  : ${INDEX}"
echo "Chunk map    : ${CHUNK_MAP}"
echo "Target run   : ${TARGET_RUN}"
echo "================================================================"
echo ""


# ---------------------------------------------------------------
# Run final merge + FV selection
# ---------------------------------------------------------------

python3 "$SCRIPT" \
    --run "$TARGET_RUN" \
    $EXTRA_ARGS

STATUS=$?


# ---------------------------------------------------------------
# Check execution status
# ---------------------------------------------------------------

if [ $STATUS -ne 0 ]; then

    echo ""
    echo "ERROR: Final merge/FV selection failed."
    echo "Sample : ${SAMPLE_NAME}"
    echo "Run    : ${TARGET_RUN}"
    echo "Index  : ${INDEX}"
    exit $STATUS

fi


# ---------------------------------------------------------------
# Finished
# ---------------------------------------------------------------

echo ""
echo "================================================================"
echo "Stage 5 task completed successfully"
echo "================================================================"
echo "Sample : ${SAMPLE_NAME}"
echo "Run    : ${TARGET_RUN}"
echo "Index  : ${INDEX}"
echo "================================================================"
