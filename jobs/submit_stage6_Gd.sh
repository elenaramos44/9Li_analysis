#!/bin/bash

#SBATCH --qos=regular
#SBATCH --job-name=Li9_stage6_Gd
#SBATCH --output=/scratch/elena/9Li/results/log/stage6_Gd_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/stage6_Gd_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=3:00:00

source /scratch/elena/setup_wcsim.sh

SCRIPT=/scratch/elena/9Li/scripts/delayed_neutron_search.py
TASK_ID=${SLURM_ARRAY_TASK_ID}

if [[ "$EXTRA_ARGS" == *"--bkg"* ]]; then
    CHUNK_MAP="/scratch/elena/9Li/results/Gd_chunk_map_bkg.pkl"
else
    CHUNK_MAP="/scratch/elena/9Li/results/Gd_chunk_map_signal.pkl"
fi

python3 "$SCRIPT" \
    --chunk-map "$CHUNK_MAP" \
    --chunk-id "$TASK_ID" \
    --fvtag FV_1 \
    --prompt-window-ns 20 \
    --prompt-skip-ms 20 \
    --trigger-window-ms 480 \
    --gap-us 5 \
    --search-us 150 \
    --window-ns 5 \
    --nhits-min 10 \
    --nhits-max 50 \
    --rms-cut-ns 2 \
    --dr-max-cm 20 \
    $EXTRA_ARGS \
    --verbose