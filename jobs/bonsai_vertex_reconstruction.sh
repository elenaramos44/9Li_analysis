#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=Li9_bonsai
#SBATCH --output=/scratch/elena/9Li/results/log/Li9_bonsai_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/Li9_bonsai_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-47

set -e  # ⬅️ FAIL FAST (important)

echo "===================================="
echo "Starting Li9 BONSAI reconstruction"
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "===================================="

# -----------------------------
# ENVIRONMENT
# -----------------------------
source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh
source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh

export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake

export WCSIM_BUILD_DIR=/scratch/elena/wcsim-install
export BONSAIDIR=/scratch/elena/bonsai

export LD_LIBRARY_PATH=$WCSIM_BUILD_DIR/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$BONSAIDIR/lib:$LD_LIBRARY_PATH

export ROOT_INCLUDE_PATH=$BONSAIDIR/bonsai:$WCSIM_BUILD_DIR/include/WCSim:$ROOT_INCLUDE_PATH

echo "Environment ready"

# -----------------------------
# PATHS
# -----------------------------
RUN=1848
IN_DIR=/scratch/elena/9Li/results/run${RUN}
OUT_DIR=/scratch/elena/9Li/results/run${RUN}/bonsai_output
SCRIPT=/scratch/elena/9Li/scripts/bonsai_vertex_reconstruction.py

mkdir -p $OUT_DIR
mkdir -p /scratch/elena/9Li/results/log

CSV_FILE=${IN_DIR}/Li9_clusters_chunk_${SLURM_ARRAY_TASK_ID}.csv

echo "Input CSV: $CSV_FILE"
echo "Output Dir: $OUT_DIR"

# -----------------------------
# RUN
# -----------------------------
python3 -u $SCRIPT \
    --csv $CSV_FILE \
    --outdir $OUT_DIR \
    --verbose

echo "Finished task $SLURM_ARRAY_TASK_ID"