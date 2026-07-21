#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=Li9_final_fv
#SBATCH --output=/scratch/elena/9Li/results/log/merge_task_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/merge_task_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=0:30:00            
#SBATCH --array=0-11              

source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh
source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh
export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

echo "WCSim environment setup ready"

SCRIPT=/scratch/elena/9Li/scripts/merge_and_fv_cut.py
INDEX=${SLURM_ARRAY_TASK_ID}

RUNS=(1928 1930 1932 1934 1935 1936 1937 1938 1939 1941 1846 1848)
TARGET_RUN=${RUNS[$INDEX]}

if [ -z "$TARGET_RUN" ]; then
    echo "Error: Array Index out of bounds."
    exit 1
fi



if [ ! -z "$1" ]; then
    EXTRA_ARGS="$1"
elif [ -z "$EXTRA_ARGS" ]; then
    EXTRA_ARGS=""
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Merge and FV Cut Selection for Run=${TARGET_RUN} (Args: $EXTRA_ARGS)"

# Ejecución robusta
python3 $SCRIPT --run $TARGET_RUN $EXTRA_ARGS

echo "Process completed successfully for Run=${TARGET_RUN}"
