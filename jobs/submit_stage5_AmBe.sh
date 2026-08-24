#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=submit_stage5_AmBe
#SBATCH --output=/scratch/elena/9Li/results/log/submit_stage5_%j.out
#SBATCH --error=/scratch/elena/9Li/results/log/submit_stage5_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00

echo "==============================================================="
echo "Submitting Stage 5 (Final Merge + Fiducial Volume Cut) [AmBe]"
echo "Time: $(date)"
echo "==============================================================="

JOBS_DIR="/scratch/elena/9Li/jobs"

# ------------------------------------------------
# Get run number
# ------------------------------------------------

if [ -z "$1" ]; then
    echo "ERROR: No run number provided."
    echo "Usage: sbatch submit_stage5_AmBe.sh <RUN_NUMBER>"
    exit 1
fi

RUN_NUMBER=$1

# ------------------------------------------------
# Determine sample type
# ------------------------------------------------

if [ "$RUN_NUMBER" -eq 2384 ]; then

    SAMPLE_TYPE="BACKGROUND"

elif [[ "$RUN_NUMBER" -eq 2387 || "$RUN_NUMBER" -eq 2388 || \
        "$RUN_NUMBER" -eq 2389 || "$RUN_NUMBER" -eq 2390 ]]; then

    SAMPLE_TYPE="SIGNAL"

else
    echo "ERROR: Unsupported AmBe run: ${RUN_NUMBER}"
    exit 1
fi

echo "Run:         ${RUN_NUMBER}"
echo "Sample type: ${SAMPLE_TYPE}"
echo "==============================================================="

# ------------------------------------------------
# Submit final FV-cut job
# ------------------------------------------------

JOB_OUT=$(sbatch \
    ${JOBS_DIR}/fv_cut_AmBe.sh \
    ${RUN_NUMBER})

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to submit Stage 5."
    exit 1
fi

JOB_ID=$(echo "$JOB_OUT" | awk '{print $4}')

echo "Stage 5 submitted successfully."
echo "Job ID: ${JOB_ID}"

echo "==============================================================="
echo "Entire AmBe pipeline has been successfully submitted!"
echo "Final stage is now running."
echo "Run: ${RUN_NUMBER}"
echo "Sample: ${SAMPLE_TYPE}"
echo "==============================================================="