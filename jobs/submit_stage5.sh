#!/bin/bash

#SBATCH --qos=regular
#SBATCH --job-name=submit_stage5
#SBATCH --output=/scratch/elena/9Li/results/log/submit_stage5_%j.out
#SBATCH --error=/scratch/elena/9Li/results/log/submit_stage5_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00

echo "==============================================================="
echo "Submitting Stage 5 (Final Merge + Fiducial Volume Cut)"
echo "Time: $(date)"
echo "==============================================================="

BASE_DIR="/scratch/elena/9Li"
JOBS_DIR="$BASE_DIR/jobs"
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
# Determine unique runs from the SAME spill-aware chunk map
# ---------------------------------------------------------------

echo ""
echo "Reading runs from spill-aware chunk map..."


TOTAL_RUNS=$(python3 - "$CHUNK_MAP" <<'PY'
import sys
import pickle

chunk_map = sys.argv[1]

with open(chunk_map, "rb") as f:
    chunks = pickle.load(f)

unique_runs = sorted(
    set(chunk["run"] for chunk in chunks)
)

print(len(unique_runs))
PY
)

STATUS=$?

if [ $STATUS -ne 0 ] || [ -z "$TOTAL_RUNS" ]; then

    echo "ERROR: Could not determine number of runs."
    exit 1

fi


if ! [[ "$TOTAL_RUNS" =~ ^[0-9]+$ ]]; then

    echo "ERROR: Invalid number of runs: '$TOTAL_RUNS'"
    exit 1

fi


if [ "$TOTAL_RUNS" -eq 0 ]; then

    echo "ERROR: Chunk map contains no runs."
    exit 1

fi


MAX_INDEX=$((TOTAL_RUNS - 1))
ARRAY_RANGE="0-${MAX_INDEX}"


echo ""
echo "---------------------------------------------------------------"
echo "Stage 5 run information"
echo "---------------------------------------------------------------"
echo "Sample       : $SAMPLE_NAME"
echo "Chunk map    : $CHUNK_MAP"
echo "Total runs   : $TOTAL_RUNS"
echo "Max index    : $MAX_INDEX"
echo "Array range  : $ARRAY_RANGE"
echo "---------------------------------------------------------------"
echo ""


# ---------------------------------------------------------------
# Submit Stage 5 array
# ---------------------------------------------------------------

echo "Submitting Stage 5 array..."


JOB_OUT=$(sbatch \
    --array="${ARRAY_RANGE}" \
    --export=ALL,EXTRA_ARGS="${SAMPLE_FLAG}" \
    "$JOBS_DIR/fv_cut.sh"
)

STATUS=$?

if [ $STATUS -ne 0 ]; then

    echo "ERROR: Failed to submit Stage 5 array."
    exit 1

fi


JOB_ID=$(echo "$JOB_OUT" | awk '{print $4}')


if [ -z "$JOB_ID" ]; then

    echo "ERROR: Could not retrieve Stage 5 Job ID."
    echo "sbatch output:"
    echo "$JOB_OUT"
    exit 1

fi


echo ""
echo "--> Stage 5 Job ID: $JOB_ID"
echo "--> Stage 5 array: $ARRAY_RANGE"
echo "--> Sample: $SAMPLE_NAME"
echo ""


# ---------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------

echo "==============================================================="
echo "Stage 5 successfully submitted"
echo "==============================================================="
echo "Sample       : $SAMPLE_NAME"
echo "Chunk map    : $CHUNK_MAP"
echo "Total runs   : $TOTAL_RUNS"
echo "Array        : $ARRAY_RANGE"
echo "Stage 5 Job  : $JOB_ID"
echo "==============================================================="