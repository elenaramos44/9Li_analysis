#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=Li9_multilat
#SBATCH --output=/scratch/elena/9Li/results/log/Li9_multilat_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/Li9_multilat_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-49


echo "Setting environment for multilateration"

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


RUN=1846

IN_DIR=/scratch/elena/9Li/results/run${RUN}
OUT_DIR=/scratch/elena/9Li/results/run${RUN}/multilat_output
SCRIPT=/scratch/elena/9Li/scripts/multilat_vertex_reconstruction.py

mkdir -p $OUT_DIR



TASK_ID=${SLURM_ARRAY_TASK_ID}
CSV_FILE=${IN_DIR}/Li9_clusters_chunk_${TASK_ID}.csv

echo "------------------------------"
echo "Run: $RUN"
echo "Chunk: $TASK_ID"
echo "Input: $CSV_FILE"
echo "Output: $OUT_DIR"
echo "------------------------------"


python3 $SCRIPT \
    --csv $CSV_FILE \
    --outdir $OUT_DIR \
    --verbose

echo "Finished chunk ${TASK_ID}"