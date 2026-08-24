#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=AmBe_refine_parallel
#SBATCH --output=/scratch/elena/9Li/results/log/refine_task_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/refine_task_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00

source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh
source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh

export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

echo "WCSim environment setup ready"

SCRIPT=/scratch/elena/9Li/scripts/refinement_all.py
TASK_ID=${SLURM_ARRAY_TASK_ID}

# ------------------------------------------------
# Get run number
# ------------------------------------------------

if [ -z "$1" ]; then
    echo "ERROR: No run number provided."
    exit 1
fi

TARGET_RUN=$1
TARGET_CHUNK=${TASK_ID}

# ------------------------------------------------
# Determine sample type
# ------------------------------------------------

if [ "$TARGET_RUN" -eq 2384 ]; then

    BKG_FLAG="--bkg"
    SAMPLE_TYPE="BACKGROUND"

elif [[ "$TARGET_RUN" -eq 2387 || "$TARGET_RUN" -eq 2388 || \
        "$TARGET_RUN" -eq 2389 || "$TARGET_RUN" -eq 2390 ]]; then

    BKG_FLAG=""
    SAMPLE_TYPE="SIGNAL"

else
    echo "ERROR: Unsupported AmBe run: ${TARGET_RUN}"
    exit 1
fi

# ------------------------------------------------
# Input/output directory
# ------------------------------------------------

IN_DIR="/scratch/elena/9Li/results/run${TARGET_RUN}/processed"
OUT_DIR="$IN_DIR"

mkdir -p "$OUT_DIR"

echo "--------------------------------------------------------"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Refinement"
echo "Sample:       $SAMPLE_TYPE"
echo "Run:          $TARGET_RUN"
echo "Chunk:        $TARGET_CHUNK"
echo "Input Dir:    $IN_DIR"
echo "Output Dir:   $OUT_DIR"
echo "--------------------------------------------------------"

# ------------------------------------------------
# Run refinement
# ------------------------------------------------

python3 $SCRIPT \
    --run $TARGET_RUN \
    --chunk-id $TARGET_CHUNK \
    $BKG_FLAG

echo "Task finished successfully: run=${TARGET_RUN} chunk=${TARGET_CHUNK}"