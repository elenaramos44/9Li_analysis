#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=AmBe_pipeline
#SBATCH --output=/scratch/elena/9Li/results/log/pipeline_AmBe_%j.out
#SBATCH --error=/scratch/elena/9Li/results/log/pipeline_AmBe_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00

echo "================================================================"
echo "Starting AmBe Pipeline (Run 2384) on Hyperion"
echo "Time: $(date)"
echo "================================================================"

JOBS_DIR="/scratch/elena/9Li/jobs"

# Array range calculated with get_chunks.py (157,955 events / 25,000 chunk size)
ARRAY_RANGE="0-6%10"

echo "Using Array Range: $ARRAY_RANGE"

# ------------------------------------------------------------------------------
# STAGE 1: Submit Load Sliding Windows
# ------------------------------------------------------------------------------

echo "Submitting Stage 1: Load sliding windows"

JOB_OUT_1=$(sbatch \
    --array=${ARRAY_RANGE} \
    ${JOBS_DIR}/load_SW_all_files_AmBe.sh) || exit 1

JOB_ID_1=$(echo "$JOB_OUT_1" | awk '{print $4}')

echo "--> Deployed Stage 1 Job ID: $JOB_ID_1"

# ------------------------------------------------------------------------------
# Submit Stage 2 launcher
# ------------------------------------------------------------------------------

echo "Submitting Stage 2 launcher"

JOB_OUT_2=$(sbatch \
    --dependency=afterok:${JOB_ID_1} \
    ${JOBS_DIR}/submit_stage2_AmBe.sh)

JOB_ID_2=$(echo "$JOB_OUT_2" | awk '{print $4}')

echo "--> Deployed Stage 2 launcher Job ID: $JOB_ID_2 (Dependent on $JOB_ID_1)"

echo "================================================================"
echo "AmBe Pipeline successfully started."
echo "The remaining stages will be submitted automatically."
echo "================================================================"
