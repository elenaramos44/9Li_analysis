#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=submit_stage2_AmBe
#SBATCH --output=/scratch/elena/9Li/results/log/submit_stage2_%j.out
#SBATCH --error=/scratch/elena/9Li/results/log/submit_stage2_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00

JOBS_DIR="/scratch/elena/9Li/jobs"

# Rango exacto de chunks para AmBe (0 a 6)
ARRAY_RANGE="0-6%10"

echo "Submitting Stage 2 for AmBe (Run 2384)..."
echo "Using Array Range: ${ARRAY_RANGE}"

JOB_OUT=$(sbatch \
    --array=${ARRAY_RANGE} \
    ${JOBS_DIR}/run_multilat_all_AmBe.sh)

JOB_ID=$(echo "$JOB_OUT" | awk '{print $4}')

echo "Stage 2 submitted with Job ID: ${JOB_ID}"

# Lanza el lanzador de Stage 4 cuando termine Stage 2
sbatch \
    --dependency=afterok:${JOB_ID} \
    ${JOBS_DIR}/submit_stage4_AmBe.sh
