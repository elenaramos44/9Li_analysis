#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=submit_stage5_Gd
#SBATCH --output=/scratch/elena/9Li/results/log/submit_stage5_Gd_%j.out
#SBATCH --error=/scratch/elena/9Li/results/log/submit_stage5_Gd_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00

echo "==============================================================="
echo "Submitting Stage 5 (Gd) (Final Merge + Fiducial Volume Cut)"
echo "Time: $(date)"
echo "==============================================================="

JOBS_DIR="/scratch/elena/9Li/jobs"

# Recuperar flags pasadas desde Stage 4 (e.g., --bkg)
SAMPLE_FLAG="${EXTRA_ARGS}"

echo "Submitting Stage 5 array for 6 Gd runs (array=0-5)..."

JOB_OUT=$(sbatch \
    --array=0-5 \
    --export=ALL,EXTRA_ARGS="${SAMPLE_FLAG}" \
    ${JOBS_DIR}/fv_cut_Gd.sh)

JOB_ID=$(echo "$JOB_OUT" | awk '{print $4}')

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to submit Stage 5 Array for Gd!"
    exit 1
fi

echo "Stage 5 (Gd) submitted successfully."
echo "Job ID: ${JOB_ID}"

echo "==============================================================="
echo "Entire 9Li Gadolinium Pipeline has been successfully submitted!"
echo "Final stage is now running."
echo "==============================================================="