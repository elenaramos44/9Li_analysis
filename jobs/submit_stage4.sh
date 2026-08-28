#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=submit_stage4
#SBATCH --output=/scratch/elena/9Li/results/log/submit_stage4_%j.out
#SBATCH --error=/scratch/elena/9Li/results/log/submit_stage4_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00

echo "==============================================================="
echo "Submitting Stage 4 (Refinement)"
echo "Time: $(date)"
echo "==============================================================="

JOBS_DIR="/scratch/elena/9Li/jobs"
BASE_DIR="/scratch/elena/9Li"
CHUNK_MAP_DIR="$BASE_DIR/chunk_maps"


# ---------------------------------------------------------------
# Recover Signal / Background mode
# ---------------------------------------------------------------

SAMPLE_FLAG="${EXTRA_ARGS}"

if [[ "$SAMPLE_FLAG" == "--bkg" ]]; then

    SAMPLE_NAME="BACKGROUND"
    CHUNK_MAP="$CHUNK_MAP_DIR/bkg_chunks.pkl"

    echo ">> BACKGROUND MODE <<"

else

    SAMPLE_NAME="SIGNAL"
    CHUNK_MAP="$CHUNK_MAP_DIR/signal_chunks.pkl"

    echo ">> SIGNAL MODE <<"

fi


echo "Sample    : $SAMPLE_NAME"
echo "Chunk map : $CHUNK_MAP"


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
# Determine number of chunks from spill-aware map
# ---------------------------------------------------------------

echo ""
echo "Reading number of chunks from spill-aware chunk map..."

TOTAL_CHUNKS=$(python3 - "$CHUNK_MAP" <<'PY'
import sys
import pickle

chunk_map = sys.argv[1]

with open(chunk_map, "rb") as f:
    chunks = pickle.load(f)

print(len(chunks))
PY
)

STATUS=$?

if [ $STATUS -ne 0 ] || [ -z "$TOTAL_CHUNKS" ]; then

    echo "ERROR: Could not determine number of chunks."
    exit 1

fi


if ! [[ "$TOTAL_CHUNKS" =~ ^[0-9]+$ ]]; then

    echo "ERROR: Invalid number of chunks: '$TOTAL_CHUNKS'"
    exit 1

fi


if [ "$TOTAL_CHUNKS" -eq 0 ]; then

    echo "ERROR: Chunk map contains zero chunks."
    exit 1

fi


MAX_INDEX=$((TOTAL_CHUNKS - 1))
ARRAY_RANGE="0-${MAX_INDEX}%10"


echo ""
echo "---------------------------------------------------------------"
echo "Stage 4 chunk information"
echo "---------------------------------------------------------------"
echo "Sample       : $SAMPLE_NAME"
echo "Chunk map    : $CHUNK_MAP"
echo "Total chunks : $TOTAL_CHUNKS"
echo "Max index    : $MAX_INDEX"
echo "Array range  : $ARRAY_RANGE"
echo "---------------------------------------------------------------"
echo ""


# ---------------------------------------------------------------
# Submit Stage 4 array
# ---------------------------------------------------------------

echo "Submitting Stage 4 array..."

JOB_OUT=$(sbatch \
    --array="${ARRAY_RANGE}" \
    --export=ALL,EXTRA_ARGS="${SAMPLE_FLAG}" \
    "$JOBS_DIR/submit_refinement.sh"
)

STATUS=$?

if [ $STATUS -ne 0 ]; then

    echo "ERROR: Failed to submit Stage 4 array."
    exit 1

fi


JOB_ID=$(echo "$JOB_OUT" | awk '{print $4}')


if [ -z "$JOB_ID" ]; then

    echo "ERROR: Could not retrieve Stage 4 Job ID."
    echo "sbatch output:"
    echo "$JOB_OUT"
    exit 1

fi


echo ""
echo "--> Stage 4 Job ID: $JOB_ID"
echo "--> Stage 4 array: $ARRAY_RANGE"
echo "--> Sample: $SAMPLE_NAME"
echo ""


# ---------------------------------------------------------------
# Stage 5 launcher
# ---------------------------------------------------------------

echo "Submitting Stage 5 launcher..."
echo "Waiting for Stage 4 Job ID: $JOB_ID"
echo ""

sbatch \
    --dependency=afterok:${JOB_ID} \
    --export=ALL,EXTRA_ARGS="${SAMPLE_FLAG}" \
    "$JOBS_DIR/submit_stage5.sh"

STATUS=$?

if [ $STATUS -ne 0 ]; then

    echo "ERROR: Failed to submit Stage 5 launcher."
    exit 1

fi


echo ""
echo "==============================================================="
echo "Stage 4 successfully submitted"
echo "==============================================================="
echo "Sample       : $SAMPLE_NAME"
echo "Chunk map    : $CHUNK_MAP"
echo "Total chunks : $TOTAL_CHUNKS"
echo "Array        : $ARRAY_RANGE"
echo "Stage 4 Job   : $JOB_ID"
echo "Stage 5       : waiting with afterok"
echo "==============================================================="