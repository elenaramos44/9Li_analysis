#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=Skim_Li9_Pions
#SBATCH --output=/scratch/elena/9Li/filtered_root/log/%A/run_%a.out
#SBATCH --error=/scratch/elena/9Li/filtered_root/log/%A/run_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --array=0-11%4           # Max 4 runs processing at the same time

mkdir -p /scratch/elena/9Li/filtered_root/log/%A

# Environment setup (Matching your workspace exactly)
source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh
source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh
export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

echo "WCSim environment setup ready for skimming"

# Define parameters and script path
SCRIPT=/scratch/elena/9Li/scripts/filter_pion_spills.py
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Simple run array matching index-to-index
RUNS=(1846 1848 1928 1930 1932 1934 1935 1936 1937 1938 1939 1941)
TARGET_RUN=${RUNS[$TASK_ID]}

if [ -z "$TARGET_RUN" ]; then
    echo "Error: TASK_ID $TASK_ID out of bounds."
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Skimming Task=${TASK_ID} -> Processing Run=${TARGET_RUN}"

# Execution block
python3 $SCRIPT \
    --run $TARGET_RUN \
    --in-base /data/elena/data \
    --out-base /scratch/elena/9Li/filtered_root

echo "Skimming Task finished successfully for run=${TARGET_RUN}"