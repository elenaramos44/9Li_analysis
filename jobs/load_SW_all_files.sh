#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=Li9_hits_multi
#SBATCH --output=/scratch/elena/9Li/results/log/task_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/task_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --array=0-564%50          #maximum of 50 running at once (ALL: 845%50)


#environment setup
source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh
source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh
export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

echo "WCSim environment setup ready"


CHUNK_SIZE=25000
SCRIPT=/scratch/elena/9Li/scripts/load_and_sliding_windows.py
TASK_ID=${SLURM_ARRAY_TASK_ID}

#from get_chunks.py

RUNS=(1928 1930 1932 1934 1935 1936 1937 1938 1939 1941 1846 1848)

PATHS=(
    "/scratch/elena/9Li/filtered_root/p_260"
    "/scratch/elena/9Li/filtered_root/p_260"
    "/scratch/elena/9Li/filtered_root/p_260"
    "/scratch/elena/9Li/filtered_root/p_260"
    "/scratch/elena/9Li/filtered_root/p_260"
    "/scratch/elena/9Li/filtered_root/p_260"
    "/scratch/elena/9Li/filtered_root/p_260"
    "/scratch/elena/9Li/filtered_root/p_260"
    "/scratch/elena/9Li/filtered_root/p_260"
    "/scratch/elena/9Li/filtered_root/p_260"
    "/scratch/elena/9Li/filtered_root/p_340"
    "/scratch/elena/9Li/filtered_root/p_340"
)

#CHUNKS_PER_RUN=(50 48 79 92 97 66 68 58 79 44 72 93)    #raw runs

#CHUNKS_PER_RUN=(24 13 34 19 22 20 27 13 21 31 33 33)     #filtered runs
CHUNKS_PER_RUN=(55 80 64 48 47 39 52 31 52 63 18 16)     #bkg


CURRENT_SUM=0
TARGET_RUN=""
TARGET_PATH=""
TARGET_CHUNK=""

for i in "${!RUNS[@]}"; do
    NUM_CHUNKS=${CHUNKS_PER_RUN[$i]}
    NEXT_SUM=$((CURRENT_SUM + NUM_CHUNKS))
    
    if [ "$TASK_ID" -lt "$NEXT_SUM" ]; then
        TARGET_RUN=${RUNS[$i]}
        TARGET_PATH=${PATHS[$i]}
        TARGET_CHUNK=$((TASK_ID - CURRENT_SUM))
        break
    fi
    CURRENT_SUM=$NEXT_SUM
done


if [ -z "$TARGET_RUN" ]; then
    echo "Error: TASK_ID $TASK_ID out of bounds."
    exit 1
fi

OUTDIR=/scratch/elena/9Li/results/run${TARGET_RUN}
mkdir -p $OUTDIR

#execution block
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Global Task=${TASK_ID} -> Processing Run=${TARGET_RUN} Chunk=${TARGET_CHUNK} Path=${TARGET_PATH}"

python3 $SCRIPT \
    --run $TARGET_RUN \
    --chunk-id $TARGET_CHUNK \
    --chunk-size $CHUNK_SIZE \
    --outdir $OUTDIR \
    --base-path $TARGET_PATH \
    --verbose

echo "Task finished successfully: run=${TARGET_RUN} chunk=${TARGET_CHUNK}"