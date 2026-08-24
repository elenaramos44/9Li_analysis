#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=NiCf_multilat
#SBATCH --output=/scratch/elena/9Li/results/log/multilat_task_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/multilat_task_%A_%a.err
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
export WCSIM_BUILD_DIR=/scratch/elena/wcsim-install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

export BONSAIDIR=/scratch/elena/bonsai
export LD_LIBRARY_PATH=$BONSAIDIR:$LD_LIBRARY_PATH
export ROOT_INCLUDE_PATH=$BONSAIDIR/bonsai:/scratch/elena/wcsim-install/include/WCSim:$ROOT_INCLUDE_PATH

SCRIPT=/scratch/elena/9Li/scripts/multilat_vertex_reconstruction.py

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
INPUT_FILE="${IN_DIR}/Li9_clusters_chunk_${TARGET_CHUNK}_BKG.pkl"

mkdir -p "$OUT_DIR"

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input PKL does not exist:"
    echo "$INPUT_FILE"
    exit 1
fi

python3 "$SCRIPT" \
    --pkl "$INPUT_FILE" \
    --outdir "$OUT_DIR" \
    --bkg \
    --verbose