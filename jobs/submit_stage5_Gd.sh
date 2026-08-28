#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=submit_stage5_Gd
#SBATCH --output=/scratch/elena/9Li/results/log/submit_stage5_Gd_%j.out
#SBATCH --error=/scratch/elena/9Li/results/log/submit_stage5_Gd_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00

echo "==============================================================="
echo "Submitting Stage 5 (Gd): Final Merge + Fiducial Volume Cut"
echo "Time: $(date)"
echo "==============================================================="

JOBS_DIR="/scratch/elena/9Li/jobs"
GD_BASE_DIR="/scratch/elena/9Li/filtered_root/Gd"

# ------------------------------------------------------------------------------
# 1. Background / signal configuration
# ------------------------------------------------------------------------------

EXTRA_FLAGS="${EXTRA_ARGS}"

SAMPLE_TYPE="signal"
if [[ "$EXTRA_FLAGS" == *"--bkg"* ]]; then
    SAMPLE_TYPE="bkg"
    echo ">> STAGE 5: GADOLINIUM BACKGROUND MODE <<"
else
    echo ">> STAGE 5: GADOLINIUM SIGNAL MODE <<"
fi

echo "Sample type : ${SAMPLE_TYPE}"

# ------------------------------------------------------------------------------
# 2. Determine the runs belonging to this sample type
# ------------------------------------------------------------------------------

echo ""
echo "Determining Gd runs for Stage 5..."

TOTAL_RUNS=$(python3 -c "
import os

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

print(len(runs))
")

STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "ERROR: Failed to determine the number of Gd runs."
    exit $STATUS
fi

if [ -z "$TOTAL_RUNS" ] || [ "$TOTAL_RUNS" -eq 0 ]; then
    echo "ERROR: No Gd ${SAMPLE_TYPE} runs found for Stage 5."
    exit 1
fi

MAX_IDX=$((TOTAL_RUNS - 1))
ARRAY_RANGE="0-${MAX_IDX}"

echo ""
echo "==============================================================="
echo "Stage 5 configuration"
echo "==============================================================="
echo "Base directory : ${GD_BASE_DIR}"
echo "Sample type    : ${SAMPLE_TYPE}"
echo "Total runs     : ${TOTAL_RUNS}"
echo "Array range    : ${ARRAY_RANGE}"
echo "==============================================================="

# ------------------------------------------------------------------------------
# 3. Submit Stage 5 array
# ------------------------------------------------------------------------------

echo ""
echo "Submitting Stage 5 array..."

JOB_OUT=$(sbatch \
    --array="${ARRAY_RANGE}" \
    --export=ALL,EXTRA_ARGS="${EXTRA_FLAGS}" \
    "${JOBS_DIR}/fv_cut_Gd.sh")

STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "ERROR: Failed to submit Stage 5 Array for Gd!"
    exit $STATUS
fi

JOB_ID=$(echo "$JOB_OUT" | awk '{print $4}')

if [ -z "$JOB_ID" ]; then
    echo "ERROR: Failed to retrieve Stage 5 Job ID."
    exit 1
fi

echo ""
echo "==============================================================="
echo "Stage 5 (Gd) submitted successfully"
echo "==============================================================="
echo "Sample type : ${SAMPLE_TYPE}"
echo "Runs        : ${TOTAL_RUNS}"
echo "Array       : ${ARRAY_RANGE}"
echo "Job ID      : ${JOB_ID}"
echo "==============================================================="

exit 0
