#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=Li9_refine_Gd
#SBATCH --output=/scratch/elena/9Li/results/log/refine_task_Gd_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/refine_task_Gd_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------

source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh
source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh

export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

echo "================================================================"
echo "WCSim environment setup ready (Refinement Gd)"
echo "Time: $(date)"
echo "================================================================"

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

SCRIPT="/scratch/elena/9Li/scripts/refinement_all.py"
TASK_ID="${SLURM_ARRAY_TASK_ID}"
TASK_MAP="${TASK_MAP}"

RESULTS_DIR="/scratch/elena/9Li/results"

# ------------------------------------------------------------------------------
# Check task map
# ------------------------------------------------------------------------------

if [ -z "$TASK_MAP" ] || [ ! -f "$TASK_MAP" ]; then
    echo "ERROR: TASK_MAP not found:"
    echo "  ${TASK_MAP}"
    exit 1
fi

echo "Using Stage 4 task map:"
echo "  ${TASK_MAP}"

echo "SLURM array task:"
echo "  ${TASK_ID}"

# ------------------------------------------------------------------------------
# Determine sample type
# ------------------------------------------------------------------------------

if [[ "${EXTRA_ARGS}" == *"--bkg"* ]]; then
    echo "Sample type: GADOLINIUM BACKGROUND"
else
    echo "Sample type: GADOLINIUM SIGNAL"
fi

# ------------------------------------------------------------------------------
# Read the task corresponding to this SLURM array index
# ------------------------------------------------------------------------------

read -r TARGET_RUN TARGET_CHUNK INPUT_FILE < <(
python3 - "${TASK_MAP}" "${TASK_ID}" <<'PY'
import pickle
import sys

task_map = sys.argv[1]
task_id = int(sys.argv[2])

with open(task_map, "rb") as f:
    tasks = pickle.load(f)

if task_id >= len(tasks):
    print("EOF EOF EOF")
    sys.exit(0)

task = tasks[task_id]

print(
    task["run"],
    task["chunk_id"],
    task["input_file"]
)
PY
)

# ------------------------------------------------------------------------------
# Check task validity
# ------------------------------------------------------------------------------

if [ "$TARGET_RUN" == "EOF" ] || [ -z "$TARGET_RUN" ]; then
    echo "Task ID ${TASK_ID} exceeds required chunks for Gd refinement."
    echo "Exiting cleanly."
    exit 0
fi

# ------------------------------------------------------------------------------
# Output directory
# ------------------------------------------------------------------------------

OUT_DIR="${RESULTS_DIR}/run${TARGET_RUN}/processed"

mkdir -p "$OUT_DIR"

# ------------------------------------------------------------------------------
# Print task information
# ------------------------------------------------------------------------------

echo "----------------------------------------------------------------"
echo "Stage 4 Refinement task"
echo "----------------------------------------------------------------"
echo "Global task ID : ${TASK_ID}"
echo "Run            : ${TARGET_RUN}"
echo "Chunk ID       : ${TARGET_CHUNK}"
echo "Input PKL      : ${INPUT_FILE}"
echo "Output Dir     : ${OUT_DIR}"
echo "----------------------------------------------------------------"

# ------------------------------------------------------------------------------
# Run refinement
# ------------------------------------------------------------------------------

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting refinement..."

python3 "$SCRIPT" \
    --run "$TARGET_RUN" \
    --chunk-id "$TARGET_CHUNK" \
    ${EXTRA_ARGS}

STATUS=$?

# ------------------------------------------------------------------------------
# Check Python exit status
# ------------------------------------------------------------------------------

if [ $STATUS -ne 0 ]; then

    echo "================================================================"
    echo "ERROR: Stage 4 refinement failed"
    echo "================================================================"
    echo "Global task ID : ${TASK_ID}"
    echo "Run            : ${TARGET_RUN}"
    echo "Chunk ID       : ${TARGET_CHUNK}"
    echo "Input PKL      : ${INPUT_FILE}"
    echo "Exit status    : ${STATUS}"
    echo "================================================================"

    exit $STATUS
fi

echo "================================================================"
echo "Stage 4 refinement task completed successfully"
echo "================================================================"
echo "Global task ID : ${TASK_ID}"
echo "Run            : ${TARGET_RUN}"
echo "Chunk ID       : ${TARGET_CHUNK}"
echo "Input PKL      : ${INPUT_FILE}"
echo "Time           : $(date)"
echo "================================================================"

exit 0