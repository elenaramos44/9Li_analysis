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
#SBATCH --time=00:15:00

echo "================================================================"
echo "Starting full Li9 analysis pipeline submission (Gd Runs)"
echo "Time: $(date)"
echo "================================================================"

JOBS_DIR="/scratch/elena/9Li/jobs"
SCRIPTS_DIR="/scratch/elena/9Li/scripts"

GD_BASE_DIR="/scratch/elena/9Li/filtered_root/Gd"
RESULTS_DIR="/scratch/elena/9Li/results"

CHUNK_SIZE=25000

# ------------------------------------------------------------------------------
# 1. Determination of sample type: SIGNAL or BACKGROUND
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
# 2. Environment
# ------------------------------------------------------------------------------

source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

echo "Conda environment:"
echo "  $CONDA_PREFIX"

# ------------------------------------------------------------------------------
# 3. Build spill-aware chunk map
# ------------------------------------------------------------------------------

CHUNK_MAP="${RESULTS_DIR}/Gd_chunk_map_${SAMPLE_TYPE}.pkl"

echo ""
echo "================================================================"
echo "Building spill-aware chunk map for Gd"
echo "================================================================"
echo "Base ROOT directory : ${GD_BASE_DIR}"
echo "Sample type         : ${SAMPLE_TYPE}"
echo "Chunk size          : ${CHUNK_SIZE}"
echo "Chunk map           : ${CHUNK_MAP}"
echo "================================================================"

python3 "${SCRIPTS_DIR}/build_spill_chunk_map.py" \
    --base-root "${GD_BASE_DIR}" \
    --suffix "_${SAMPLE_TYPE}.root" \
    --chunk-size "${CHUNK_SIZE}" \
    --output "${CHUNK_MAP}"

STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "ERROR: Failed to build Gd spill-aware chunk map."
    exit $STATUS
fi

if [ ! -f "${CHUNK_MAP}" ]; then
    echo "ERROR: Chunk map was not created:"
    echo "  ${CHUNK_MAP}"
    exit 1
fi

# ------------------------------------------------------------------------------
# 4. Get the REAL number of chunks from the chunk map
# ------------------------------------------------------------------------------

echo ""
echo "Calculating total chunks from spill-aware chunk map..."

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
echo "Gd chunk map ready"
echo "================================================================"
echo "Sample type       : ${SAMPLE_TYPE}"
echo "Chunk map         : ${CHUNK_MAP}"
echo "Total chunks      : ${TOTAL_CHUNKS}"
echo "SLURM array       : ${ARRAY_RANGE}"
echo "================================================================"

EXTRA_FLAGS="${SAMPLE_FLAG}"

# ------------------------------------------------------------------------------
# STAGE 1: Load sliding windows
# ------------------------------------------------------------------------------

echo ""
echo "Submitting Stage 1: Load sliding windows (Gd)..."

JOB_OUT_1=$(sbatch \
    --array="${ARRAY_RANGE}" \
    --export=ALL,EXTRA_ARGS="${EXTRA_FLAGS}",CHUNK_MAP="${CHUNK_MAP}" \
    "${JOBS_DIR}/load_SW_all_files_Gd.sh")

STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "ERROR: Failed to submit Stage 1 Array for Gd!"
    exit $STATUS
fi

JOB_ID_1=$(echo "$JOB_OUT_1" | awk '{print $4}')

if [ -z "$JOB_ID_1" ]; then
    echo "ERROR: Failed to retrieve Stage 1 Job ID."
    exit 1
fi

echo "--> Deployed Stage 1 Job ID (Gd): ${JOB_ID_1}"

# ------------------------------------------------------------------------------
# STAGE 2: Submit Stage 2 launcher
#
# IMPORTANT:
# Stage 2 uses AFTEROK, so it will only start if the entire Stage 1
# job array finishes successfully.
# ------------------------------------------------------------------------------

echo ""
echo "Submitting Stage 2 launcher..."
echo "Waiting for successful completion of Stage 1 Job ID: ${JOB_ID_1}"

JOB_OUT_2=$(sbatch \
    --dependency="afterok:${JOB_ID_1}" \
    --export=ALL,EXTRA_ARGS="${EXTRA_FLAGS}",CHUNK_MAP="${CHUNK_MAP}" \
    "${JOBS_DIR}/submit_stage2_Gd.sh")

STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "ERROR: Failed to submit Stage 2 launcher for Gd!"
    exit $STATUS
fi

JOB_ID_2=$(echo "$JOB_OUT_2" | awk '{print $4}')

if [ -z "$JOB_ID_2" ]; then
    echo "ERROR: Failed to retrieve Stage 2 Job ID."
    exit 1
fi

echo "--> Deployed Stage 2 launcher (Gd): ${JOB_ID_2}"
echo "    Dependency: afterok:${JOB_ID_1}"

# ------------------------------------------------------------------------------
# Final summary
# ------------------------------------------------------------------------------

echo ""
echo "================================================================"
echo "Gd Pipeline successfully started."
echo "================================================================"
echo "Sample type : ${SAMPLE_TYPE}"
echo "Chunk map   : ${CHUNK_MAP}"
echo "Chunks      : ${TOTAL_CHUNKS}"
echo "Array       : ${ARRAY_RANGE}"
echo "Stage 1     : ${JOB_ID_1}"
echo "Stage 2     : ${JOB_ID_2}"
echo ""
echo "Stage 2 will start ONLY if Stage 1 completes successfully."
echo "The remaining stages will be submitted automatically."
echo "================================================================"

exit 0