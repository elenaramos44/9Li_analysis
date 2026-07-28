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

echo "Submitting Stage 5 job for AmBe Run 2384..."

# Lanzamos el trabajo de merge para el run 2384
JOB_OUT=$(sbatch ${JOBS_DIR}/fv_cut_AmBe.sh)

JOB_ID=$(echo "$JOB_OUT" | awk '{print $4}')

echo "Stage 5 submitted successfully."
echo "Job ID: ${JOB_ID}"

echo "==============================================================="
echo "Entire AmBe pipeline has been successfully submitted!"
echo "Final stage is now running."
echo "==============================================================="
