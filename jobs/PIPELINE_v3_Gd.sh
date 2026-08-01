#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=Li9_pipeline_Gd
#SBATCH --output=/scratch/elena/9Li/results/log/pipeline_Gd_%j.out
#SBATCH --error=/scratch/elena/9Li/results/log/pipeline_Gd_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

echo "================================================================"
echo "Starting full Li9 analysis pipeline submission (Gd Runs) on Hyperion"
echo "Time: $(date)"
echo "================================================================"

JOBS_DIR="/scratch/elena/9Li/jobs"
GD_DIR="/scratch/elena/9Li/filtered_root/Gd/p_270"
CHUNK_SIZE=25000

# ------------------------------------------------------------------------------
# 1. Determinación del tipo de muestra (Signal o Background)
# ------------------------------------------------------------------------------
SAMPLE_TYPE="signal"
SAMPLE_FLAG=""

if [[ "$1" == "--bkg" ]]; then
    SAMPLE_TYPE="bkg"
    SAMPLE_FLAG="--bkg"
    echo ">> CONFIGURATION SET TO: GADOLINIUM BACKGROUND (BKG) <<"
else
    echo ">> CONFIGURATION SET TO: GADOLINIUM SIGNAL (DEFAULT) <<"
fi

# ------------------------------------------------------------------------------
# 2. Cálculo dinámico del rango del Slurm Array usando uproot
# ------------------------------------------------------------------------------
echo "Calculating total chunks for Gd (${SAMPLE_TYPE}) in ${GD_DIR}..."

source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

TOTAL_CHUNKS=$(python3 -c "
import uproot, glob, math, os

files = glob.glob('${GD_DIR}/*_${SAMPLE_TYPE}.root')
total_entries = 0

for f in files:
    try:
        with uproot.open(f) as root_file:
            total_entries += root_file['WCTEReadoutWindows'].num_entries
    except Exception as e:
        print(f'Warning: Could not read {f}: {e}', flush=True)

chunks = math.ceil(total_entries / ${CHUNK_SIZE})
print(chunks)
")

if [ -z "$TOTAL_CHUNKS" ] || [ "$TOTAL_CHUNKS" -eq 0 ]; then
    echo "Error: No valid ROOT files found or total chunks calculated is 0."
    exit 1
fi

MAX_INDEX=$((TOTAL_CHUNKS - 1))
ARRAY_RANGE="0-${MAX_INDEX}%10"

echo "--> Total Chunks calculated (size=${CHUNK_SIZE}): ${TOTAL_CHUNKS}"
echo "--> Configured SLURM Array Range: ${ARRAY_RANGE}"

# Únicamente pasamos --bkg si aplica, sin inventos raros
EXTRA_FLAGS="${SAMPLE_FLAG}"

# ------------------------------------------------------------------------------
# STAGE 1: Submit Load Sliding Windows (Gd)
# ------------------------------------------------------------------------------
echo "Submitting Stage 1: Load sliding windows (Gd)..."

JOB_OUT_1=$(sbatch \
    --array=${ARRAY_RANGE} \
    --export=ALL,EXTRA_ARGS="${EXTRA_FLAGS}" \
    ${JOBS_DIR}/load_SW_all_files_Gd.sh) || exit 1

JOB_ID_1=$(echo "$JOB_OUT_1" | awk '{print $4}')

if [ -z "$JOB_ID_1" ]; then
    echo "Error: Failed to submit Stage 1 Array for Gd!"
    exit 1
fi

echo "--> Deployed Stage 1 Job ID (Gd): $JOB_ID_1"

# ------------------------------------------------------------------------------
# STAGE 2: Submit Stage 2 launcher (Gd)
# ------------------------------------------------------------------------------
echo "Submitting Stage 2 launcher (waiting for Stage 1 Job ID: $JOB_ID_1)..."

JOB_OUT_2=$(sbatch \
    --dependency=afterany:${JOB_ID_1} \
    --export=ALL,EXTRA_ARGS="${EXTRA_FLAGS}" \
    ${JOBS_DIR}/submit_stage2_Gd.sh)

JOB_ID_2=$(echo "$JOB_OUT_2" | awk '{print $4}')

echo "--> Deployed Stage 2 launcher Job ID (Gd): $JOB_ID_2 (Dependent on $JOB_ID_1)"

echo "================================================================"
echo "Gd Pipeline successfully started."
echo "The remaining stages will be submitted automatically."
echo "================================================================"