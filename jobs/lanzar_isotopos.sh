#!/usr/bin/env bash
#SBATCH --qos=regular
#SBATCH --job-name=WCTE_isotopes_all
#SBATCH --output=/scratch/elena/9Li/results/log/isotopes_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/isotopes_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --array=0-11     # 12 archivos en total (del índice 0 al 11)

echo "Setting environment for WCTE isotope calculation"

source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh

echo "Environment ready."

LIST_FILE="/scratch/elena/9Li/scripts/file_list.txt"
OUT_DIR="/scratch/elena/9Li/results/isotopes_output"
SCRIPT_PATH="/scratch/elena/9Li/scripts/process_run.py"

mkdir -p $OUT_DIR
mkdir -p /scratch/elena/9Li/results/log

# Sumamos 1 al ID de Slurm porque las líneas de un fichero de texto se indexan desde 1, no desde 0
LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 1))

# Extraemos la ruta exacta de la línea correspondiente
FILE_TO_PROCESS=$(sed -n "${LINE_NUMBER}p" "$LIST_FILE")

echo "------------------------------------------------"
echo "Job ID: ${SLURM_ARRAY_JOB_ID} | Task ID (Index): ${SLURM_ARRAY_TASK_ID}"
echo "Line Number in List: ${LINE_NUMBER}"
echo "Processing File: ${FILE_TO_PROCESS}"
echo "Output Directory: ${OUT_DIR}"
echo "------------------------------------------------"

# Ejecutamos el script de Python con el archivo inequívoco
python3 $SCRIPT_PATH \
    --input "$FILE_TO_PROCESS" \
    --outdir "$OUT_DIR"

echo "Task ${SLURM_ARRAY_TASK_ID} finished."