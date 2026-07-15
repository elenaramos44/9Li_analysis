#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=submit_stage2
#SBATCH --output=/scratch/elena/9Li/results/log/submit_stage2_%j.out
#SBATCH --error=/scratch/elena/9Li/results/log/submit_stage2_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00

JOBS_DIR="/scratch/elena/9Li/jobs"

# Recover the sample type (signal/background)
SAMPLE_FLAG="${EXTRA_ARGS}"

if [[ "$SAMPLE_FLAG" == "--bkg" ]]; then
    ARRAY_RANGE="0-564%10"
else
    ARRAY_RANGE="0-287%10"
fi

echo "Submitting Stage 2..."

JOB_OUT=$(sbatch \
    --array=${ARRAY_RANGE} \
    --export=ALL,EXTRA_ARGS="${SAMPLE_FLAG}" \
    ${JOBS_DIR}/run_multilat_all.sh)

JOB_ID=$(echo "$JOB_OUT" | awk '{print $4}')

echo "Stage 2 submitted with Job ID: ${JOB_ID}"

# Launch the next launcher, but only after Stage 2 finishes
sbatch \
    --dependency=afterok:${JOB_ID} \
    --export=ALL,EXTRA_ARGS="${SAMPLE_FLAG}" \
    ${JOBS_DIR}/submit_stage4.sh