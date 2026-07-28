#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=AmBe_multilat
#SBATCH --output=/scratch/elena/9Li/results/log/multilat_task_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/multilat_task_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00

echo "Setting environment for multilateration (AmBe Run 2384)"

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

echo "Environment ready (multilateration)"

SCRIPT=/scratch/elena/9Li/scripts/multilat_vertex_reconstruction.py
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Run 2384 fijo
TARGET_RUN=2384
TARGET_CHUNK=${TASK_ID}

IN_DIR="/scratch/elena/9Li/results/run${TARGET_RUN}/processed"
OUT_DIR="/scratch/elena/9Li/results/run${TARGET_RUN}/processed"

# Nombre exacto generado por load_and_sliding_windows.py con la bandera --bkg
INPUT_FILE="${IN_DIR}/Li9_clusters_chunk_${TARGET_CHUNK}_BKG.pkl"

mkdir -p $OUT_DIR

echo "--------------------------------------------------------"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Global Task: $TASK_ID"
echo "Processing Run: $TARGET_RUN"
echo "Processing Chunk: $TARGET_CHUNK"
echo "Input PKL: $INPUT_FILE"
echo "Output Dir: $OUT_DIR"
echo "--------------------------------------------------------"

python3 $SCRIPT \
    --pkl $INPUT_FILE \
    --outdir $OUT_DIR \
    --bkg \
    --verbose

echo "Finished multilateration for chunk ${TARGET_CHUNK} of run ${TARGET_RUN}"