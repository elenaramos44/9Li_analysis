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

echo "==============================================================="
echo "Submitting Stage 4 (Refinement) for AmBe (Run 2384)"
echo "Time: $(date)"
echo "==============================================================="

JOBS_DIR="/scratch/elena/9Li/jobs"

# Rango exacto de chunks para AmBe (0 a 6)
ARRAY_RANGE="0-6%10"

echo "Submitting Stage 4 array with Range: ${ARRAY_RANGE}..."

JOB_OUT=$(sbatch \
    --array=${ARRAY_RANGE} \
    ${JOBS_DIR}/submit_refinement_AmBe.sh)

JOB_ID=$(echo "$JOB_OUT" | awk '{print $4}')

echo "Stage 4 submitted successfully."
echo "Job ID: ${JOB_ID}"

echo "Submitting Stage 5 launcher..."

# Lanza el siguiente launcher (Stage 5) cuando termine Stage 4
sbatch \
    --dependency=afterok:${JOB_ID} \
    ${JOBS_DIR}/submit_stage5_AmBe.sh

echo "Done."
