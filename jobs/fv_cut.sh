#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=Li9_final_fv
#SBATCH --output=/scratch/elena/9Li/results/log/%A/merge_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/%A/merge_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=0:30:00            # Shortened runtime since we removed plotting overhead
#SBATCH --array=0-11              # 12 parallel indices mapped to your 12 target runs

mkdir -p /scratch/elena/9Li/results/log/%A

# Environment Setup
source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh
source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh
export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

echo "WCSim environment setup ready"

SCRIPT=/scratch/elena/9Li/scripts/merge_and_fv_cut.py
INDEX=${SLURM_ARRAY_TASK_ID}

# Mapping absolute array index to specific target runs
RUNS=(1846 1848 1928 1930 1932 1934 1935 1936 1937 1938 1939 1941)
TARGET_RUN=${RUNS[$INDEX]}

if [ -z "$TARGET_RUN" ]; then
    echo "Error: Array Index out of bounds."
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Merge and FV Cut Selection for Run=${TARGET_RUN}"

# Execute calculation script
python3 $SCRIPT --run $TARGET_RUN

echo "Process completed successfully for Run=${TARGET_RUN}"