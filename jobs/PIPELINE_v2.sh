#!/bin/bash

#SBATCH --qos=regular
#SBATCH --job-name=Li9_pipeline
#SBATCH --output=/scratch/elena/9Li/results/log/pipeline_%j.out
#SBATCH --error=/scratch/elena/9Li/results/log/pipeline_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

echo "================================================================"
echo "Starting full Li9 analysis pipeline submission on Hyperion"
echo "Time: $(date)"
echo "================================================================"

BASE_DIR="/scratch/elena/9Li"
JOBS_DIR="$BASE_DIR/jobs"
SCRIPTS_DIR="$BASE_DIR/scripts"
BASE_ROOT="$BASE_DIR/filtered_root"
CHUNK_MAP_DIR="$BASE_DIR/chunk_maps"

CHUNK_SIZE=25000

# ---------------------------------------------------------------
# Create required directories
# ---------------------------------------------------------------

mkdir -p "$CHUNK_MAP_DIR"
mkdir -p "$BASE_DIR/results/log"

# ---------------------------------------------------------------
# Signal / Background configuration
# ---------------------------------------------------------------

if [[ "$1" == "--bkg" ]]; then

    SAMPLE_FLAG="--bkg"
    SAMPLE_SUFFIX="_bkg.root"
    CHUNK_MAP="$CHUNK_MAP_DIR/bkg_chunks.pkl"
    SAMPLE_NAME="BACKGROUND"

    echo ">> CONFIGURATION SET TO: BACKGROUND (BKG) <<"

elif [[ -z "$1" ]]; then

    SAMPLE_FLAG=""
    SAMPLE_SUFFIX="_signal.root"
    CHUNK_MAP="$CHUNK_MAP_DIR/signal_chunks.pkl"
    SAMPLE_NAME="SIGNAL"

    echo ">> CONFIGURATION SET TO: SIGNAL <<"

else

    echo "ERROR: Unknown argument '$1'"
    echo ""
    echo "Usage:"
    echo "  sbatch PIPELINE_v2.sh"
    echo "  sbatch PIPELINE_v2.sh --bkg"
    exit 1

fi

echo ""
echo "Sample type       : $SAMPLE_NAME"
echo "Sample suffix     : $SAMPLE_SUFFIX"
echo "Chunk size target : $CHUNK_SIZE"
echo "Chunk map         : $CHUNK_MAP"
echo ""

# ---------------------------------------------------------------
# Check that the chunk-map builder exists
# ---------------------------------------------------------------

CHUNK_BUILDER="$SCRIPTS_DIR/build_spill_chunk_map.py"

if [ ! -f "$CHUNK_BUILDER" ]; then

    echo "ERROR: Spill-aware chunk map builder not found:"
    echo "$CHUNK_BUILDER"
    echo ""
    echo "Please create this script before running the pipeline."
    exit 1

fi

# ---------------------------------------------------------------
# Build spill-aware chunk map
# ---------------------------------------------------------------
#
# IMPORTANT:
#
# The chunk builder must guarantee that a spill_counter is NEVER
# split between two chunks.
#
# CHUNK_SIZE is therefore a TARGET size, not a hard upper limit.
#
# If adding a complete spill would exceed CHUNK_SIZE, that entire
# spill is moved to the next chunk.
#
# ---------------------------------------------------------------

echo "================================================================"
echo "Building spill-aware chunk map"
echo "================================================================"

python3 "$CHUNK_BUILDER" \
    --base-root "$BASE_ROOT" \
    --suffix "$SAMPLE_SUFFIX" \
    --chunk-size "$CHUNK_SIZE" \
    --output "$CHUNK_MAP"

STATUS=$?

if [ $STATUS -ne 0 ]; then

    echo ""
    echo "ERROR: Failed to build spill-aware chunk map."
    echo "Chunk map: $CHUNK_MAP"
    exit 1

fi

if [ ! -f "$CHUNK_MAP" ]; then

    echo ""
    echo "ERROR: Chunk map was not created:"
    echo "$CHUNK_MAP"
    exit 1

fi

echo ""
echo "Chunk map successfully created:"
echo "$CHUNK_MAP"

# ---------------------------------------------------------------
# Determine number of chunks from the chunk map
# ---------------------------------------------------------------

echo ""
echo "Reading number of chunks from chunk map..."

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

    echo "ERROR: No chunks found for $SAMPLE_NAME."
    exit 1

fi

MAX_INDEX=$((TOTAL_CHUNKS - 1))
ARRAY_RANGE="0-${MAX_INDEX}%10"

echo ""
echo "---------------------------------------------------------------"
echo "Spill-aware chunk information"
echo "---------------------------------------------------------------"
echo "Sample       : $SAMPLE_NAME"
echo "Chunk target : $CHUNK_SIZE entries"
echo "Total chunks : $TOTAL_CHUNKS"
echo "Max index    : $MAX_INDEX"
echo "Array range  : $ARRAY_RANGE"
echo "---------------------------------------------------------------"
echo ""

# ---------------------------------------------------------------
# STAGE 1
# Submit Load Sliding Windows
# ---------------------------------------------------------------

echo "================================================================"
echo "Submitting Stage 1: Load sliding windows"
echo "================================================================"

JOB_OUT_1=$(sbatch \
    --array="${ARRAY_RANGE}" \
    --export=ALL,EXTRA_ARGS="${SAMPLE_FLAG}" \
    "$JOBS_DIR/load_SW_all_files.sh")

STATUS=$?

if [ $STATUS -ne 0 ]; then

    echo "ERROR: Failed to submit Stage 1 array."
    exit 1

fi

JOB_ID_1=$(echo "$JOB_OUT_1" | awk '{print $4}')

if [ -z "$JOB_ID_1" ]; then

    echo "ERROR: Could not retrieve Stage 1 Job ID."
    echo "sbatch output:"
    echo "$JOB_OUT_1"
    exit 1

fi

echo ""
echo "--> Stage 1 Job ID: $JOB_ID_1"
echo "--> Stage 1 array: $ARRAY_RANGE"
echo "--> Sample: $SAMPLE_NAME"
echo ""

# ---------------------------------------------------------------
# STAGE 2
# ---------------------------------------------------------------

echo "================================================================"
echo "Submitting Stage 2 launcher"
echo "================================================================"

echo "Waiting for Stage 1 Job ID: $JOB_ID_1"
echo ""

JOB_OUT_2=$(sbatch \
    --dependency=afterok:${JOB_ID_1} \
    --export=ALL,EXTRA_ARGS="${SAMPLE_FLAG}" \
    "$JOBS_DIR/submit_stage2.sh")

STATUS=$?

if [ $STATUS -ne 0 ]; then

    echo "ERROR: Failed to submit Stage 2 launcher."
    exit 1

fi

JOB_ID_2=$(echo "$JOB_OUT_2" | awk '{print $4}')

if [ -z "$JOB_ID_2" ]; then

    echo "ERROR: Could not retrieve Stage 2 Job ID."
    echo "sbatch output:"
    echo "$JOB_OUT_2"
    exit 1

fi

echo "--> Stage 2 Job ID: $JOB_ID_2"
echo "--> Dependency: afterok:$JOB_ID_1"
echo ""

# ---------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------

echo "================================================================"
echo "Pipeline successfully submitted"
echo "================================================================"
echo "Sample type       : $SAMPLE_NAME"
echo "Sample suffix     : $SAMPLE_SUFFIX"
echo "Chunk size target : $CHUNK_SIZE"
echo "Total chunks      : $TOTAL_CHUNKS"
echo "Array             : $ARRAY_RANGE"
echo "Chunk map         : $CHUNK_MAP"
echo "Stage 1 Job ID    : $JOB_ID_1"
echo "Stage 2 Job ID    : $JOB_ID_2"
echo "================================================================"
echo "Time: $(date)"
echo "================================================================"