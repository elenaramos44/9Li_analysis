#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=NiCf_hits
#SBATCH --output=/scratch/elena/9Li/results/log/task_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/task_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00

source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh
source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh

export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

CHUNK_SIZE=25000
SCRIPT=/scratch/elena/9Li/scripts/load_and_sliding_windows_NiCf.py

TARGET_RUN=$1
TARGET_CHUNK=${SLURM_ARRAY_TASK_ID}

if [ -z "$TARGET_RUN" ]; then
    echo "ERROR: No run number provided."
    exit 1
fi

case "$TARGET_RUN" in
    2437|2482|2494|2504|2507|2508)
        ;;
    *)
        echo "ERROR: Unsupported NiCf run: ${TARGET_RUN}"
        exit 1
        ;;
esac

TARGET_PATH="/scratch/elena/9Li/filtered_root/NiCf_bkg"
OUTDIR="/scratch/elena/9Li/results/run${TARGET_RUN}/processed"

mkdir -p "$OUTDIR"

python3 "$SCRIPT" \
    --run "$TARGET_RUN" \
    --chunk-id "$TARGET_CHUNK" \
    --chunk-size "$CHUNK_SIZE" \
    --outdir "$OUTDIR" \
    --base-path "$TARGET_PATH" \
    --verbose