#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=Skim_Pions_Gd
#SBATCH --output=/scratch/elena/9Li/filtered_root/log/Skim_Gd_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/filtered_root/log/Skim_Gd_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --array=0-5%4           # 6 runs (0 a 5), max 4 a la vez

source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh
source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh
export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

echo "WCSim environment setup ready"

SCRIPT=/scratch/elena/9Li/scripts/filter_pion_spills_Gd.py
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Configuración para runs con Gd
RUNS=(2407 2408 2409 2432 2434 2438)

TARGET_RUN=${RUNS[$TASK_ID]}

if [ -z "$TARGET_RUN" ]; then
    echo "Error: TASK_ID $TASK_ID out of bounds."
    exit 1
fi

# Creamos la carpeta de salida para Gd si no existe
mkdir -p /scratch/elena/9Li/filtered_root/Gd

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Skimming Task=${TASK_ID} -> Processing Run=${TARGET_RUN}"

# Execution block
python3 $SCRIPT \
    --run $TARGET_RUN \
    --in-base /data/elena/data \
    --out-base /scratch/elena/9Li/filtered_root

echo "Skimming Task finished successfully for run=${TARGET_RUN}"