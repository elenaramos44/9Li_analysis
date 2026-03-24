#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=preproc_1848
#SBATCH --output=/scratch/elena/WCTE_DATA_ANALYSIS/9Li/pre_processing/run1848/logs/preproc_1848_%A_%a.out
#SBATCH --error=/scratch/elena/WCTE_DATA_ANALYSIS/9Li/pre_processing/run1848/logs/preproc_1848_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-69   # Ajusta según TOTAL_ROOT_FILES / FILES_PER_JOB

module purge
module load foss/2019b
module load Python/3.7.4-GCCcore-8.3.0
source /scratch/elena/elena_wcsim/build/env_wcsim.sh
export PYTHONPATH=/scratch/$USER/python-libs:$PYTHONPATH

RUN=1848
SCRIPT=/scratch/elena/9Li/scripts/pre_processing.py
DATA_DIR=/scratch/elena/WCTE_DATA_ANALYSIS/9Li/data
OUTDIR=/scratch/elena/WCTE_DATA_ANALYSIS/pre_processing/run${RUN}/npz
mkdir -p $OUTDIR

# ---------------- Block of files ----------------
FILES_PER_JOB=5
START=$((SLURM_ARRAY_TASK_ID * FILES_PER_JOB + 1))
END=$((START + FILES_PER_JOB - 1))
TOTAL=$(ls $DATA_DIR/WCTE_offline_R${RUN}S0P*.root | wc -l)
if [ $END -gt $TOTAL ]; then END=$TOTAL; fi

ROOT_FILES=$(ls $DATA_DIR/WCTE_offline_R${RUN}S0P*.root | sort | sed -n "${START},${END}p")

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing files $START to $END"

python3 $SCRIPT --root-files $ROOT_FILES --outdir $OUTDIR

echo "Task finished: files $START-$END"