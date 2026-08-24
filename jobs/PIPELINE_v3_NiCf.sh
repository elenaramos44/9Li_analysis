#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=NiCf_pipeline
#SBATCH --output=/scratch/elena/9Li/results/log/pipeline_NiCf_%j.out
#SBATCH --error=/scratch/elena/9Li/results/log/pipeline_NiCf_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00

# ================================================================
# NiCf Pipeline
#
# Usage:
#   sbatch PIPELINE_v3_NiCf.sh 2437
#   sbatch PIPELINE_v3_NiCf.sh 2482
#   sbatch PIPELINE_v3_NiCf.sh 2494
#   sbatch PIPELINE_v3_NiCf.sh 2504
#   sbatch PIPELINE_v3_NiCf.sh 2507
#   sbatch PIPELINE_v3_NiCf.sh 2508
#
# All NiCf runs are background (beam-off) samples.
# ================================================================

JOBS_DIR="/scratch/elena/9Li/jobs"

# ------------------------------------------------
# Get run number
# ------------------------------------------------

if [ -z "$1" ]; then
    echo "ERROR: No run number provided."
    echo ""
    echo "Usage:"
    echo "  sbatch PIPELINE_v3_NiCf.sh <RUN_NUMBER>"
    echo ""
    echo "Allowed runs:"
    echo "  2437 2482 2494 2504 2507 2508"
    exit 1
fi

RUN_NUMBER=$1

# ------------------------------------------------
# Check run number
# ------------------------------------------------

case "$RUN_NUMBER" in
    2437|2482|2494|2504|2507|2508)
        ;;
    *)
        echo "ERROR: Unsupported NiCf run: ${RUN_NUMBER}"
        echo "Allowed runs: 2437 2482 2494 2504 2507 2508"
        exit 1
        ;;
esac

# ------------------------------------------------
# Input
# ------------------------------------------------

INPUT_FILE="/scratch/elena/9Li/filtered_root/NiCf_bkg/WCTE_merged_production_R${RUN_NUMBER}_bkg.root"

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input ROOT file does not exist:"
    echo "$INPUT_FILE"
    exit 1
fi

# ------------------------------------------------
# Calculate number of chunks
# ------------------------------------------------

CHUNK_SIZE=25000

echo "================================================================"
echo "Starting NiCf Pipeline"
echo "================================================================"
echo "Run:          ${RUN_NUMBER}"
echo "Sample type:  BACKGROUND"
echo "Input file:   ${INPUT_FILE}"
echo "Chunk size:   ${CHUNK_SIZE}"
echo "Time:         $(date)"
echo "================================================================"

CHUNK_INFO=$(python3 /scratch/elena/9Li/scripts/get_chunks.py \
    "$INPUT_FILE" \
    --size "$CHUNK_SIZE")

echo "$CHUNK_INFO"

MAX_ARRAY_INDEX=$(echo "$CHUNK_INFO" | \
    awk '/SLURM setting:/ {
        match($0, /0-[0-9]+/)
        if (RSTART) {
            range = substr($0, RSTART, RLENGTH)
            split(range, a, "-")
            print a[2]
        }
    }')

if [ -z "$MAX_ARRAY_INDEX" ]; then
    echo "ERROR: Could not determine SLURM array range."
    exit 1
fi

ARRAY_RANGE="0-${MAX_ARRAY_INDEX}%10"

# ------------------------------------------------
# Stage 1
# ------------------------------------------------

echo "================================================================"
echo "Submitting Stage 1: Load sliding windows"
echo "================================================================"

JOB_OUT_1=$(sbatch \
    --array=${ARRAY_RANGE} \
    ${JOBS_DIR}/load_SW_all_files_NiCf.sh \
    ${RUN_NUMBER})

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to submit Stage 1."
    exit 1
fi

JOB_ID_1=$(echo "$JOB_OUT_1" | awk '{print $4}')

echo "--> Stage 1 Job ID: $JOB_ID_1"

# ------------------------------------------------
# Stage 2 launcher
# ------------------------------------------------

echo "================================================================"
echo "Submitting Stage 2 launcher"
echo "================================================================"

JOB_OUT_2=$(sbatch \
    --dependency=afterok:${JOB_ID_1} \
    ${JOBS_DIR}/submit_stage2_NiCf.sh \
    ${RUN_NUMBER})

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to submit Stage 2 launcher."
    exit 1
fi

JOB_ID_2=$(echo "$JOB_OUT_2" | awk '{print $4}')

echo "--> Stage 2 launcher Job ID: $JOB_ID_2"

echo "================================================================"
echo "NiCf Pipeline successfully started."
echo "Run: ${RUN_NUMBER}"
echo "Stage 1 Job ID: ${JOB_ID_1}"
echo "Stage 2 launcher Job ID: ${JOB_ID_2}"
echo "================================================================"