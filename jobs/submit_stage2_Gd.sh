#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=submit_stage2_Gd
#SBATCH --output=/scratch/elena/9Li/results/log/submit_stage2_Gd_%j.out
#SBATCH --error=/scratch/elena/9Li/results/log/submit_stage2_Gd_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00

echo "================================================================"
echo "Starting Stage 2 launcher (Gd)"
echo "Time: $(date)"
echo "================================================================"

JOBS_DIR="/scratch/elena/9Li/jobs"
RESULTS_DIR="/scratch/elena/9Li/results"

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------

source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

echo "Conda environment:"
echo "  $CONDA_PREFIX"

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

CHUNK_SIZE=25000
CHUNK_MAP="${CHUNK_MAP}"
EXTRA_FLAGS="${EXTRA_ARGS}"

# ------------------------------------------------------------------------------
# Check CHUNK_MAP
# ------------------------------------------------------------------------------

if [ -z "$CHUNK_MAP" ] || [ ! -f "$CHUNK_MAP" ]; then
    echo "ERROR: CHUNK_MAP not found."
    echo "  CHUNK_MAP = ${CHUNK_MAP}"
    exit 1
fi

echo "Using spill-aware chunk map:"
echo "  ${CHUNK_MAP}"

# ------------------------------------------------------------------------------
# Determine sample type
# ------------------------------------------------------------------------------

if [[ "$EXTRA_FLAGS" == *"--bkg"* ]]; then
    SAMPLE_TYPE="bkg"
    echo ">> STAGE 2 LAUNCHER: GADOLINIUM BACKGROUND MODE <<"
else
    SAMPLE_TYPE="signal"
    echo ">> STAGE 2 LAUNCHER: GADOLINIUM SIGNAL MODE <<"
fi

# ------------------------------------------------------------------------------
# Get the REAL number of chunks from the SAME chunk map used in Stage 1
# ------------------------------------------------------------------------------

echo ""
echo "Reading total number of Stage 2 tasks from chunk map..."

TOTAL_CHUNKS=$(python3 -c "
import pickle

path = '${CHUNK_MAP}'

with open(path, 'rb') as f:
    chunks = pickle.load(f)

print(len(chunks))
")

STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "ERROR: Failed to read chunk map."
    exit $STATUS
fi

if [ -z "$TOTAL_CHUNKS" ] || [ "$TOTAL_CHUNKS" -eq 0 ]; then
    echo "ERROR: Chunk map contains zero chunks."
    exit 1
fi

MAX_INDEX=$((TOTAL_CHUNKS - 1))

# Maximum of 10 simultaneous tasks
ARRAY_RANGE="0-${MAX_INDEX}%10"

echo ""
echo "================================================================"
echo "Stage 2 configuration"
echo "================================================================"
echo "Sample type       : ${SAMPLE_TYPE}"
echo "Chunk map         : ${CHUNK_MAP}"
echo "Total chunks      : ${TOTAL_CHUNKS}"
echo "SLURM array       : ${ARRAY_RANGE}"
echo "================================================================"

# ------------------------------------------------------------------------------
# STAGE 2: Submit multilateration array
# ------------------------------------------------------------------------------

echo ""
echo "Submitting Stage 2: Multilateration array (Gd)..."

JOB_OUT=$(sbatch \
    --array="${ARRAY_RANGE}" \
    --export=ALL,EXTRA_ARGS="${EXTRA_FLAGS}",CHUNK_MAP="${CHUNK_MAP}" \
    "${JOBS_DIR}/run_multilat_all_Gd.sh")

STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "ERROR: Failed to submit Stage 2 array for Gd."
    exit $STATUS
fi

JOB_ID=$(echo "$JOB_OUT" | awk '{print $4}')

if [ -z "$JOB_ID" ]; then
    echo "ERROR: Failed to retrieve Stage 2 Job ID."
    exit 1
fi

echo "--> Stage 2 array submitted."
echo "    Job ID   : ${JOB_ID}"
echo "    Array    : ${ARRAY_RANGE}"

# ------------------------------------------------------------------------------
# STAGE 4: Submit Stage 4 launcher
#
# IMPORTANT:
# afterok means Stage 4 launcher will ONLY start if the COMPLETE
# Stage 2 array finishes successfully.
# ------------------------------------------------------------------------------

echo ""
echo "Submitting Stage 4 launcher..."
echo "Waiting for successful completion of Stage 2 Job ID: ${JOB_ID}"

JOB_OUT_4=$(sbatch \
    --dependency="afterok:${JOB_ID}" \
    --export=ALL,EXTRA_ARGS="${EXTRA_FLAGS}",CHUNK_MAP="${CHUNK_MAP}" \
    "${JOBS_DIR}/submit_stage4_Gd.sh")

STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "ERROR: Failed to submit Stage 4 launcher for Gd."
    exit $STATUS
fi

JOB_ID_4=$(echo "$JOB_OUT_4" | awk '{print $4}')

if [ -z "$JOB_ID_4" ]; then
    echo "ERROR: Failed to retrieve Stage 4 launcher Job ID."
    exit 1
fi

echo "--> Stage 4 launcher submitted."
echo "    Job ID    : ${JOB_ID_4}"
echo "    Dependency: afterok:${JOB_ID}"

# ------------------------------------------------------------------------------
# Final summary
# ------------------------------------------------------------------------------

echo ""
echo "================================================================"
echo "Stage 2 launcher completed successfully."
echo "================================================================"
echo "Sample type : ${SAMPLE_TYPE}"
echo "Chunk map   : ${CHUNK_MAP}"
echo "Chunks      : ${TOTAL_CHUNKS}"
echo "Stage 2     : ${JOB_ID}"
echo "Stage 4     : ${JOB_ID_4}"
echo ""
echo "Stage 4 will start ONLY if the complete Stage 2 array"
echo "finishes successfully."
echo "================================================================"

exit 0