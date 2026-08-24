#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=submit_stage4_AmBe
#SBATCH --output=/scratch/elena/9Li/results/log/submit_stage4_%j.out
#SBATCH --error=/scratch/elena/9Li/results/log/submit_stage4_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00

JOBS_DIR="/scratch/elena/9Li/jobs"

# ------------------------------------------------
# Get run number
# ------------------------------------------------

if [ -z "$1" ]; then
    echo "ERROR: No run number provided."
    echo "Usage: sbatch submit_stage4_AmBe.sh <RUN_NUMBER>"
    exit 1
fi

RUN_NUMBER=$1

# ------------------------------------------------
# Determine sample type
# ------------------------------------------------

if [ "$RUN_NUMBER" -eq 2384 ]; then

    SAMPLE_TYPE="BKG"

elif [[ "$RUN_NUMBER" -eq 2387 || "$RUN_NUMBER" -eq 2388 || \
        "$RUN_NUMBER" -eq 2389 || "$RUN_NUMBER" -eq 2390 ]]; then

    SAMPLE_TYPE="SIGNAL"

else
    echo "ERROR: Unsupported AmBe run: ${RUN_NUMBER}"
    exit 1
fi

# ------------------------------------------------
# Input file
# ------------------------------------------------

CHUNK_SIZE=25000

if [ "$RUN_NUMBER" -eq 2384 ]; then
    INPUT_FILE="/scratch/elena/9Li/filtered_root/AmBe_bkg/WCTE_merged_production_R${RUN_NUMBER}_bkg.root"
else
    INPUT_FILE="/scratch/elena/9Li/filtered_root/AmBe_sig/WCTE_merged_production_R${RUN_NUMBER}_signal.root"
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input ROOT file does not exist:"
    echo "$INPUT_FILE"
    exit 1
fi

echo "==============================================================="
echo "Submitting Stage 4 (Refinement)"
echo "==============================================================="
echo "Run:          ${RUN_NUMBER}"
echo "Sample type:  ${SAMPLE_TYPE}"
echo "Input ROOT:   ${INPUT_FILE}"
echo "Chunk size:   ${CHUNK_SIZE}"
echo "Time:         $(date)"
echo "==============================================================="

# ------------------------------------------------
# Calculate number of chunks
# ------------------------------------------------

CHUNK_INFO=$(python3 /scratch/elena/9Li/scripts/get_chunks.py \
    "$INPUT_FILE" \
    --size "$CHUNK_SIZE")

echo "$CHUNK_INFO"

# Extract the maximum array index from:
# SLURM setting: --array=0-4
MAX_ARRAY_INDEX=$(echo "$CHUNK_INFO" | \
    sed -n 's/.*--array=0-\([0-9]\+\).*/\1/p')

if [ -z "$MAX_ARRAY_INDEX" ]; then
    echo "ERROR: Could not determine SLURM array range."
    exit 1
fi

ARRAY_RANGE="0-${MAX_ARRAY_INDEX}%10"

echo "Using Array Range: ${ARRAY_RANGE}"

# ------------------------------------------------
# Submit Stage 4 array
# ------------------------------------------------

JOB_OUT=$(sbatch \
    --array="${ARRAY_RANGE}" \
    "${JOBS_DIR}/submit_refinement_AmBe.sh" \
    "${RUN_NUMBER}")

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to submit Stage 4."
    exit 1
fi

JOB_ID=$(echo "$JOB_OUT" | awk '{print $4}')

echo "Stage 4 submitted successfully."
echo "Job ID: ${JOB_ID}"

# ------------------------------------------------
# Submit Stage 5 launcher
# ------------------------------------------------

echo "Submitting Stage 5 launcher..."

JOB_OUT_5=$(sbatch \
    --dependency=afterok:${JOB_ID} \
    "${JOBS_DIR}/submit_stage5_AmBe.sh" \
    "${RUN_NUMBER}")

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to submit Stage 5."
    exit 1
fi

JOB_ID_5=$(echo "$JOB_OUT_5" | awk '{print $4}')

echo "Stage 5 launcher submitted successfully."
echo "Stage 5 Job ID: ${JOB_ID_5}"

echo "Done."