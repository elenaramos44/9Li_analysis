#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=AmBe_hits_multi
#SBATCH --output=/scratch/elena/9Li/results/log/task_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/task_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# Cargar entorno
source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh
source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh
export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

CHUNK_SIZE=25000
SCRIPT=/scratch/elena/9Li/scripts/load_and_sliding_windows.py
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Configuración para todos los runs de AmBe (2384=bkg run, 2387, 2388, 2389 and 2390 a signal runs)
# ------------------------------------------------
# Get run number from PIPELINE_v3_AmBe.sh
# ------------------------------------------------

if [ -z "$1" ]; then
    echo "ERROR: No run number provided."
    exit 1
fi

TARGET_RUN=$1
TARGET_CHUNK=${SLURM_ARRAY_TASK_ID}

# ------------------------------------------------
# Determine sample type and input path
# ------------------------------------------------

if [ "$TARGET_RUN" -eq 2384 ]; then

    # AmBe background run
    TARGET_PATH="/scratch/elena/9Li/filtered_root/AmBe_bkg"
    BKG_FLAG="--bkg"
    SAMPLE_TYPE="BKG"

elif [[ "$TARGET_RUN" -eq 2387 || "$TARGET_RUN" -eq 2388 || \
        "$TARGET_RUN" -eq 2389 || "$TARGET_RUN" -eq 2390 ]]; then

    # AmBe signal runs
    TARGET_PATH="/scratch/elena/9Li/filtered_root/AmBe_sig"
    BKG_FLAG=""
    SAMPLE_TYPE="SIGNAL"

else
    echo "ERROR: Unsupported AmBe run: ${TARGET_RUN}"
    exit 1
fi

# ------------------------------------------------
# Output directory
# ------------------------------------------------

OUTDIR="/scratch/elena/9Li/results/run${TARGET_RUN}/processed"
mkdir -p "$OUTDIR"

echo "================================================================"
echo "Processing AmBe ${SAMPLE_TYPE}"
echo "Run:       ${TARGET_RUN}"
echo "Chunk:     ${TARGET_CHUNK}"
echo "Path:      ${TARGET_PATH}"
echo "Outdir:    ${OUTDIR}"
echo "================================================================"

# ------------------------------------------------
# Run load_and_sliding_windows.py
# ------------------------------------------------

python3 $SCRIPT \
    --run $TARGET_RUN \
    --chunk-id $TARGET_CHUNK \
    --chunk-size $CHUNK_SIZE \
    --outdir $OUTDIR \
    --base-path $TARGET_PATH \
    $BKG_FLAG \
    --verbose

echo "Task finished successfully: run=${TARGET_RUN} chunk=${TARGET_CHUNK}"