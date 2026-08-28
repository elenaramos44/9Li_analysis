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
#SBATCH --time=4:30:00

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------

source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh
conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh
source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh

export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

echo "================================================================"
echo "WCSim environment setup ready (Gd Stage 5)"
echo "Time: $(date)"
echo "================================================================"

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

SCRIPT="/scratch/elena/9Li/scripts/merge_and_fv_cut.py"
INDEX="${SLURM_ARRAY_TASK_ID}"
GD_BASE_DIR="/scratch/elena/9Li/filtered_root/Gd"

EXTRA_FLAGS="${EXTRA_ARGS}"

SAMPLE_TYPE="signal"

if [[ "$EXTRA_FLAGS" == *"--bkg"* ]]; then
    SAMPLE_TYPE="bkg"
    echo "Sample type: GADOLINIUM BACKGROUND"
else
    echo "Sample type: GADOLINIUM SIGNAL"
fi

echo "Array index : ${INDEX}"
echo "Sample type : ${SAMPLE_TYPE}"

# ------------------------------------------------------------------------------
# Determine the run corresponding to this array index
# ------------------------------------------------------------------------------

TARGET_RUN=$(python3 -c "
import os
import sys

base_dir = '${GD_BASE_DIR}'
suffix = '_${SAMPLE_TYPE}.root'
runs = set()

for sub in ['p_270', 'p_350']:
    d = os.path.join(base_dir, sub)

    if not os.path.isdir(d):
        continue

    for f in os.listdir(d):
        if (
            f.startswith('WCTE_merged_production_R')
            and f.endswith(suffix)
        ):
            run_str = (
                f.replace('WCTE_merged_production_R', '')
                 .replace(suffix, '')
            )

            try:
                runs.add(int(run_str))
            except ValueError:
                pass

runs = sorted(runs)

idx = ${INDEX}

if idx < len(runs):
    print(runs[idx])
else:
    print('EOF')
")

STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "ERROR: Failed to determine target Gd run."
    exit $STATUS
fi

# ------------------------------------------------------------------------------
# Validate target run
# ------------------------------------------------------------------------------

if [ "$TARGET_RUN" == "EOF" ] || [ -z "$TARGET_RUN" ]; then
    echo "ERROR: Array index ${INDEX} is out of bounds for Gd ${SAMPLE_TYPE} runs."
    exit 1
fi

echo ""
echo "================================================================"
echo "Starting Stage 5"
echo "================================================================"
echo "Global array index : ${INDEX}"
echo "Sample type        : ${SAMPLE_TYPE}"
echo "Target run         : ${TARGET_RUN}"
echo "================================================================"

# ------------------------------------------------------------------------------
# Run final merge + fiducial-volume selection
# ------------------------------------------------------------------------------

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting merge_and_fv_cut.py"

python3 "$SCRIPT" \
    --run "$TARGET_RUN" \
    $EXTRA_FLAGS

STATUS=$?

# ------------------------------------------------------------------------------
# Check Python exit status
# ------------------------------------------------------------------------------

if [ $STATUS -ne 0 ]; then
    echo "================================================================"
    echo "ERROR: Stage 5 failed"
    echo "================================================================"
    echo "Array index : ${INDEX}"
    echo "Run         : ${TARGET_RUN}"
    echo "Sample type : ${SAMPLE_TYPE}"
    echo "Exit status : ${STATUS}"
    echo "================================================================"

    exit $STATUS
fi

echo "================================================================"
echo "Stage 5 task completed successfully"
echo "================================================================"
echo "Array index : ${INDEX}"
echo "Run         : ${TARGET_RUN}"
echo "Sample type : ${SAMPLE_TYPE}"
echo "Time        : $(date)"
echo "================================================================"

exit 0
