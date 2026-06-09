#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=Li9_to_pkl
#SBATCH --output=/scratch/elena/9Li/results/log/pkl_task_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/pkl_task_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --array=0-11    # 12 runs

echo "Setting environment for PKL conversion"
source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env


RUNS=(1928 1930 1932 1934 1935 1936 1937 1938 1939 1941 1846 1848 )

TARGET_RUN=${RUNS[${SLURM_ARRAY_TASK_ID}]}

SCRIPT=/scratch/elena/9Li/scripts/procesado_csv_multilat.py
OUT_DIR=/scratch/elena/9Li/results/run${TARGET_RUN}/processed

echo "--------------------------------------------------------"
echo "Global Task: ${SLURM_ARRAY_TASK_ID} -> Processing Entire Run: $TARGET_RUN"
echo "Output Directory: $OUT_DIR"
echo "--------------------------------------------------------"


python3 $SCRIPT \
    --run $TARGET_RUN \
    --outdir $OUT_DIR

echo "Finished conversion completely for run ${TARGET_RUN}"