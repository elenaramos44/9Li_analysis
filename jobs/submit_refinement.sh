#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=Li9_refine_parallel
#SBATCH --output=/scratch/elena/9Li/results/log/%A/task_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/%A/task_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4         # Reduced from 8 to optimize resource allocation, as chunks are processed line-by-line
#SBATCH --mem=8G                 # Reduced from 16G since chunks are loaded individually, saving memory pool space
#SBATCH --time=2:00:00           # Reduced walltime; processing 1 individual chunk per slot is much faster
#SBATCH --array=0-845%50         # Tracks 846 total chunks across all runs simultaneously

mkdir -p /scratch/elena/9Li/results/log/%A

# Environment Setup
source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh
source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh
export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

echo "WCSim environment setup ready"

SCRIPT=/scratch/elena/9Li/scripts/refinement_all.py
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Lists mapped for linear array decoding
RUNS=(1846 1848 1928 1930 1932 1934 1935 1936 1937 1938 1939 1941)
CHUNKS_PER_RUN=(50 48 79 92 97 66 68 58 79 44 72 93)

CURRENT_SUM=0
TARGET_RUN=""
TARGET_CHUNK=""

# Iterate and slice the absolute index down to Run and local Chunk ID 
for i in "${!RUNS[@]}"; do
    NUM_CHUNKS=${CHUNKS_PER_RUN[$i]}
    NEXT_SUM=$((CURRENT_SUM + NUM_CHUNKS))
    
    if [ "$TASK_ID" -lt "$NEXT_SUM" ]; then
        TARGET_RUN=${RUNS[$i]}
        TARGET_CHUNK=$((TASK_ID - CURRENT_SUM))
        break
    fi
    CURRENT_SUM=$NEXT_SUM
done

if [ -z "$TARGET_RUN" ]; then
    echo "Error: TASK_ID $TASK_ID out of boundaries."
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Global Task=${TASK_ID} -> Starting Refinement for Run=${TARGET_RUN} Chunk=${TARGET_CHUNK}"

# Execute python execution per array process task
python3 $SCRIPT \
    --run $TARGET_RUN \
    --chunk-id $TARGET_CHUNK

echo "Task finished successfully: run=${TARGET_RUN} chunk=${TARGET_CHUNK}"