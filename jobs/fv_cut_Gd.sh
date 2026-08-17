#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=Li9_final_fv_Gd
#SBATCH --output=/scratch/elena/9Li/results/log/merge_task_Gd_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/merge_task_Gd_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=0:30:00

source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh
source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh
export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

echo "WCSim environment setup ready (Stage 5 Gd)"

# Captura las variables pasadas por export
EXTRA_ARGS=${1:-$EXTRA_ARGS}

SCRIPT=/scratch/elena/9Li/scripts/merge_and_fv_cut.py
INDEX=${SLURM_ARRAY_TASK_ID}

# Lista de 6 runs de Gadolinio
#RUNS=(2407 2408 2409 2432 2434 2438)
RUNS=(2374 2379)

TARGET_RUN=${RUNS[$INDEX]}

if [ -z "$TARGET_RUN" ]; then
    echo "Error: Array Index out of bounds for Gd runs."
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Merge and FV Cut Selection for Gd Run=${TARGET_RUN} (Args: $EXTRA_ARGS)"

# Ejecución del script Python original agnóstico
python3 $SCRIPT --run $TARGET_RUN $EXTRA_ARGS

echo "Process completed successfully for Gd Run=${TARGET_RUN}"