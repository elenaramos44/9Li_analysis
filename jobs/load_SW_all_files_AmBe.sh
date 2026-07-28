#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=AmBe_hits_multi
#SBATCH --output=/scratch/elena/9Li/results/log/task_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/task_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# Cargar entorno
source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh
source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh
export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

CHUNK_SIZE=25000
SCRIPT=/scratch/elena/9Li/scripts/load_and_sliding_windows.py
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Configuración única para AmBe (Run 2384)
TARGET_RUN=2384
TARGET_PATH="/scratch/elena/9Li/filtered_root/AmBe_bkg"
TARGET_CHUNK=${TASK_ID}

# Directorio de salida dedicado para el Run 2384
OUTDIR=/scratch/elena/9Li/results/run${TARGET_RUN}/processed
mkdir -p $OUTDIR

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing AmBe Run=${TARGET_RUN} Chunk=${TARGET_CHUNK} Path=${TARGET_PATH}"

# Forzamos la bandera --bkg para que lea el archivo WCTE_merged_production_R2384_bkg.root
python3 $SCRIPT \
    --run $TARGET_RUN \
    --chunk-id $TARGET_CHUNK \
    --chunk-size $CHUNK_SIZE \
    --outdir $OUTDIR \
    --base-path $TARGET_PATH \
    --bkg \
    --verbose

echo "Task finished successfully: run=${TARGET_RUN} chunk=${TARGET_CHUNK}"