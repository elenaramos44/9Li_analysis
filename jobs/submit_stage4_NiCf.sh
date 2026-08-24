#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=submit_stage4_NiCf
#SBATCH --output=/scratch/elena/9Li/results/log/submit_stage4_%j.out
#SBATCH --error=/scratch/elena/9Li/results/log/submit_stage4_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00

JOBS_DIR="/scratch/elena/9Li/jobs"

if [ -z "$1" ]; then
    echo "ERROR: No run number provided."
    exit 1
fi

RUN_NUMBER=$1

case "$RUN_NUMBER" in
    2437|2482|2494|2504|2507|2508)
        ;;
    *)
        echo "ERROR: Unsupported NiCf run: ${RUN_NUMBER}"
        exit 1
        ;;
esac

INPUT_FILE="/scratch/elena/9Li/filtered_root/NiCf_bkg/WCTE_merged_production_R${RUN_NUMBER}_bkg.root"

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input ROOT file does not exist:"
    echo "$INPUT_FILE"
    exit 1
fi

CHUNK_SIZE=25000

CHUNK_INFO=$(python3 /scratch/elena/9Li/scripts/get_chunks.py \
    "$INPUT_FILE" \
    --size "$CHUNK_SIZE")

echo "$CHUNK_INFO"

MAX_ARRAY_INDEX=$(echo "$CHUNK_INFO" | \
    sed -n 's/.*--array=0-\([0-9]\+\).*/\1/p')

if [ -z "$MAX_ARRAY_INDEX" ]; then
    echo "ERROR: Could not determine SLURM array range."
    exit 1
fi

ARRAY_RANGE="0-${MAX_ARRAY_INDEX}%10"

echo "Submitting Stage 4 for NiCf run ${RUN_NUMBER}"
echo "Array: ${ARRAY_RANGE}"

JOB_OUT=$(sbatch \
    --array="${ARRAY_RANGE}" \
    "${JOBS_DIR}/submit_refinement_NiCf.sh" \
    "${RUN_NUMBER}")

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to submit Stage 4."
    exit 1
fi

JOB_ID=$(echo "$JOB_OUT" | awk '{print $4}')

echo "Stage 4 submitted: ${JOB_ID}"

JOB_OUT_5=$(sbatch \
    --dependency=afterok:${JOB_ID} \
    "${JOBS_DIR}/submit_stage5_NiCf.sh" \
    "${RUN_NUMBER}")

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to submit Stage 5 launcher."
    exit 1
fi

JOB_ID_5=$(echo "$JOB_OUT_5" | awk '{print $4}')

echo "Stage 5 launcher submitted: ${JOB_ID_5}"