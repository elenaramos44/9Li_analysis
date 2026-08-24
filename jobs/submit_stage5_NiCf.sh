#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=submit_stage5_NiCf
#SBATCH --output=/scratch/elena/9Li/results/log/submit_stage5_%j.out
#SBATCH --error=/scratch/elena/9Li/results/log/submit_stage5_%j.err
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

echo "Submitting Stage 5 for NiCf run ${RUN_NUMBER}"

JOB_OUT=$(sbatch \
    "${JOBS_DIR}/fv_cut_NiCf.sh" \
    "${RUN_NUMBER}")

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to submit Stage 5."
    exit 1
fi

JOB_ID=$(echo "$JOB_OUT" | awk '{print $4}')

echo "Stage 5 submitted: ${JOB_ID}"