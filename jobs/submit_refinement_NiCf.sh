#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=NiCf_refine_parallel
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

SCRIPT=/scratch/elena/9Li/scripts/refinement_all.py

if [ -z "$1" ]; then
    echo "ERROR: No run number provided."
    exit 1
fi

TARGET_RUN=$1
TARGET_CHUNK=${SLURM_ARRAY_TASK_ID}

case "$TARGET_RUN" in
    2437|2482|2494|2504|2507|2508)
        ;;
    *)
        echo "ERROR: Unsupported NiCf run: ${TARGET_RUN}"
        exit 1
        ;;
esac

IN_DIR="/scratch/elena/9Li/results/run${TARGET_RUN}/processed"
OUT_DIR="$IN_DIR"

mkdir -p "$OUT_DIR"

python3 "$SCRIPT" \
    --run "$TARGET_RUN" \
    --chunk-id "$TARGET_CHUNK" \
    --bkg