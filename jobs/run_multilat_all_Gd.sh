#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=Li9_multilat_Gd
#SBATCH --output=/scratch/elena/9Li/results/log/multilat_task_Gd_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/multilat_task_Gd_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00

echo "================================================================"
echo "Starting Gd multilateration task"
echo "Time: $(date)"
echo "================================================================"

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------

source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh
source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh

export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake
export WCSIM_BUILD_DIR=/scratch/elena/wcsim-install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

export BONSAIDIR=/scratch/elena/bonsai
export LD_LIBRARY_PATH=$BONSAIDIR:$LD_LIBRARY_PATH
export ROOT_INCLUDE_PATH=$BONSAIDIR/bonsai:/scratch/elena/wcsim-install/include/WCSim:$ROOT_INCLUDE_PATH

echo "Environment ready (multilateration Gd)"

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

SCRIPT="/scratch/elena/9Li/scripts/multilat_vertex_reconstruction.py"
RESULTS_DIR="/scratch/elena/9Li/results"

TASK_ID="${SLURM_ARRAY_TASK_ID}"
CHUNK_MAP="${CHUNK_MAP}"

# ------------------------------------------------------------------------------
# Check CHUNK_MAP
# ------------------------------------------------------------------------------

if [ -z "$CHUNK_MAP" ] || [ ! -f "$CHUNK_MAP" ]; then
    echo "ERROR: CHUNK_MAP not found."
    echo "  CHUNK_MAP = ${CHUNK_MAP}"
    exit 1
fi

echo "Using chunk map:"
echo "  ${CHUNK_MAP}"

echo "SLURM array task:"
echo "  ${TASK_ID}"

# ------------------------------------------------------------------------------
# Determine sample type
# ------------------------------------------------------------------------------

if [[ "$EXTRA_ARGS" == *"--bkg"* ]]; then
    SAMPLE_TYPE="bkg"
    echo "Sample type: GADOLINIUM BACKGROUND"
else
    SAMPLE_TYPE="signal"
    echo "Sample type: GADOLINIUM SIGNAL"
fi

# ------------------------------------------------------------------------------
# Read run and chunk information from the SAME global chunk map
# used by Stage 1
# ------------------------------------------------------------------------------

CHUNK_INFO=$(python3 -c "
import pickle
import sys

chunk_map = '${CHUNK_MAP}'
task_id = ${TASK_ID}

with open(chunk_map, 'rb') as f:
    chunks = pickle.load(f)

if task_id >= len(chunks):
    print('EOF EOF')
    sys.exit(0)

chunk = chunks[task_id]

print(
    chunk['run'],
    chunk['chunk_id']
)
")

STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "ERROR: Failed to read chunk map."
    exit $STATUS
fi

read -r TARGET_RUN TARGET_CHUNK <<< "$CHUNK_INFO"

# ------------------------------------------------------------------------------
# Check whether this task corresponds to a valid chunk
# ------------------------------------------------------------------------------

if [ "$TARGET_RUN" == "EOF" ] || [ -z "$TARGET_RUN" ]; then
    echo "Task ID ${TASK_ID} exceeds total chunks in Gd chunk map."
    echo "Exiting cleanly."
    exit 0
fi

# ------------------------------------------------------------------------------
# Determine input PKL
# ------------------------------------------------------------------------------

OUT_DIR="${RESULTS_DIR}/run${TARGET_RUN}/processed"

if [ "$SAMPLE_TYPE" == "bkg" ]; then
    INPUT_FILE="${OUT_DIR}/Li9_clusters_chunk_${TARGET_CHUNK}_BKG.pkl"
else
    INPUT_FILE="${OUT_DIR}/Li9_clusters_chunk_${TARGET_CHUNK}.pkl"
fi

# ------------------------------------------------------------------------------
# Check input PKL
# ------------------------------------------------------------------------------

if [ ! -f "$INPUT_FILE" ]; then
    echo "================================================================"
    echo "ERROR: Expected Stage 1 PKL does not exist."
    echo "================================================================"
    echo "Global task ID : ${TASK_ID}"
    echo "Run            : ${TARGET_RUN}"
    echo "Chunk ID       : ${TARGET_CHUNK}"
    echo "Expected PKL   : ${INPUT_FILE}"
    echo "================================================================"

    exit 1
fi

# ------------------------------------------------------------------------------
# Print chunk information
# ------------------------------------------------------------------------------

echo "----------------------------------------------------------------"
echo "Chunk information"
echo "----------------------------------------------------------------"
echo "Global task ID : ${TASK_ID}"
echo "Run            : ${TARGET_RUN}"
echo "Chunk ID       : ${TARGET_CHUNK}"
echo "Input PKL      : ${INPUT_FILE}"
echo "Output Dir     : ${OUT_DIR}"
echo "----------------------------------------------------------------"

mkdir -p "$OUT_DIR"

# ------------------------------------------------------------------------------
# Run multilateration
# ------------------------------------------------------------------------------

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting multilateration..."

python3 "$SCRIPT" \
    --pkl "$INPUT_FILE" \
    --outdir "$OUT_DIR" \
    $EXTRA_ARGS \
    --verbose

STATUS=$?

# ------------------------------------------------------------------------------
# Check Python exit status
# ------------------------------------------------------------------------------

if [ $STATUS -ne 0 ]; then

    echo "================================================================"
    echo "ERROR: Multilateration failed"
    echo "================================================================"
    echo "Global task ID : ${TASK_ID}"
    echo "Run            : ${TARGET_RUN}"
    echo "Chunk ID       : ${TARGET_CHUNK}"
    echo "Input PKL      : ${INPUT_FILE}"
    echo "Exit status    : ${STATUS}"
    echo "================================================================"

    exit $STATUS
fi

# ------------------------------------------------------------------------------
# Successful completion
# ------------------------------------------------------------------------------

echo "================================================================"
echo "Stage 2 task completed successfully"
echo "================================================================"
echo "Global task ID : ${TASK_ID}"
echo "Run            : ${TARGET_RUN}"
echo "Chunk ID       : ${TARGET_CHUNK}"
echo "Input PKL      : ${INPUT_FILE}"
echo "Time           : $(date)"
echo "================================================================"

exit 0