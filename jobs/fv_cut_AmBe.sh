#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=AmBe_final_fv
#SBATCH --output=/scratch/elena/9Li/results/log/merge_task_%j.out
#SBATCH --error=/scratch/elena/9Li/results/log/merge_task_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00

source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh
source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh
export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

echo "WCSim environment setup ready"

SCRIPT=/scratch/elena/9Li/scripts/merge_and_fv_cut.py

# Configuración fija para AmBe
TARGET_RUN=2384

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Merge and FV Cut Selection for AmBe Run=${TARGET_RUN}"

# Ejecutamos el script de merge pasando --run 2384 y --bkg
python3 $SCRIPT --run $TARGET_RUN --bkg

echo "Process completed successfully for Run=${TARGET_RUN}"
