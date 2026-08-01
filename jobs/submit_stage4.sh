#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=submit_stage4
#SBATCH --output=/scratch/elena/9Li/results/log/submit_stage4_%j.out
#SBATCH --error=/scratch/elena/9Li/results/log/submit_stage4_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00

echo "==============================================================="
echo "Submitting Stage 4 (Refinement)"
echo "Time: $(date)"
echo "==============================================================="

JOBS_DIR="/scratch/elena/9Li/jobs"

# Recover Signal/Background mode
SAMPLE_FLAG="${EXTRA_ARGS}"

if [[ "$SAMPLE_FLAG" == "--bkg" ]]; then
    ARRAY_RANGE="0-564%10"
    echo ">> BACKGROUND MODE <<"
else
    ARRAY_RANGE="0-287%10"
    echo ">> SIGNAL MODE <<"
fi

echo "Submitting Stage 4 array..."

JOB_OUT=$(sbatch \
    --array=${ARRAY_RANGE} \
    --export=ALL,EXTRA_ARGS="${SAMPLE_FLAG}" \
    ${JOBS_DIR}/submit_refinement.sh)

JOB_ID=$(echo "$JOB_OUT" | awk '{print $4}')

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to submit Stage 4 Array!"
    exit 1
fi

echo "Stage 4 submitted successfully."
echo "Job ID: ${JOB_ID}"

echo "Submitting Stage 5 launcher (waiting for Stage 4 Job ID: ${JOB_ID})..."

sbatch \
    --dependency=afterany:${JOB_ID} \
    --export=ALL,EXTRA_ARGS="${SAMPLE_FLAG}" \
    ${JOBS_DIR}/submit_stage5.sh

echo "==============================================================="
echo "Stage 4 and Stage 5 chain successfully submitted!"
echo "==============================================================="