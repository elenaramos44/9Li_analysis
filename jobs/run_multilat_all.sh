#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=Li9_multilat
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00

# Ejecuta los 846 chunks en total, manteniendo un máximo de 50 simultáneos
#SBATCH --array=0-845%50

# Organiza los logs de salida en una carpeta con el ID del Job
#SBATCH --output=/scratch/elena/9Li/results/log/%A/multilat_task_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/%A/multilat_task_%a.err

# Crear el directorio de logs usando la variable de entorno real de Bash
mkdir -p /scratch/elena/9Li/results/log/${SLURM_ARRAY_JOB_ID}

echo "Setting environment for multilateration"

source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh
source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh

export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake
export WCSIM_BUILD_DIR=/scratch/elena/wcsim-install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

export BONSAIDIR=/scratch/elena/bonsai
export LD_LIBRARY_PATH=$BONSAIDIR:$LD_LIBRARY_PATH
export ROOT_INCLUDE_PATH=$BONSAIDIR/bonsai:/scratch/elena/wcsim-install/include/WCSim:$ROOT_INCLUDE_PATH

echo "Environment ready (multilateration)"

# Global configurations
SCRIPT=/scratch/elena/9Li/scripts/multilat_vertex_reconstruction.py
TASK_ID=${SLURM_ARRAY_TASK_ID}

# ==============================================================================
# LÓGICA DE MAPEO EXACTA (Idéntica al script anterior)
# ==============================================================================
RUNS=(1846 1848 1928 1930 1932 1934 1935 1936 1937 1938 1939 1941)
CHUNKS_PER_RUN=(50 48 79 92 97 66 68 58 79 44 72 93) 

CURRENT_SUM=0
TARGET_RUN=""
TARGET_CHUNK=""

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

# Control de seguridad de límites
if [ -z "$TARGET_RUN" ]; then
    echo "Error: TASK_ID $TASK_ID fuera de los límites calculados."
    exit 1
fi


IN_DIR=/scratch/elena/9Li/results/run${TARGET_RUN}
OUT_DIR=/scratch/elena/9Li/results/run${TARGET_RUN}/multilat_output
CSV_FILE="${IN_DIR}/Li9_clusters_range(15-50)_chunk_${TARGET_CHUNK}.csv"

# Crea la carpeta de salida si no existe (ej. /results/run1937/multilat_output/)
mkdir -p $OUT_DIR

echo "--------------------------------------------------------"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Global Task: $TASK_ID"
echo "Processing Run: $TARGET_RUN"
echo "Processing Chunk: $TARGET_CHUNK"
echo "Input CSV: $CSV_FILE"
echo "Output Dir: $OUT_DIR"
echo "--------------------------------------------------------"

# Ejecución del script de Python
python3 $SCRIPT \
    --csv $CSV_FILE \
    --outdir $OUT_DIR \
    --verbose

echo "Finished chunk ${TARGET_CHUNK} for run ${TARGET_RUN}"