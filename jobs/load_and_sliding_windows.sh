#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=Li9_hits
#SBATCH --output=/scratch/elena/9Li/results/log/Li9_hits_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/Li9_hits_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-47       #python3 /scratch/elena/9Li/scripts/get_chunks.py /data/elena/data/WCTE_offline_R{RUN}S0_VME_matched.root


source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh
source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh
export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

echo "WCSim environment setup ready"


RUN=1848
CHUNK_SIZE=25000
OUTDIR=/scratch/elena/9Li/results/run${RUN}
SCRIPT=/scratch/elena/9Li/scripts/load_and_sliding_windows.py
BASE_PATH=/data/elena/data

mkdir -p $OUTDIR

# -------------------------------
# SLURM task info
TASK_ID=${SLURM_ARRAY_TASK_ID}
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running run=${RUN} chunk=${TASK_ID}"


python3 $SCRIPT \
    --run $RUN \
    --chunk-id $TASK_ID \
    --chunk-size $CHUNK_SIZE \
    --outdir $OUTDIR \
    --base-path $BASE_PATH \
    --verbose

echo "Task finished: run=${RUN} chunk=${TASK_ID}"