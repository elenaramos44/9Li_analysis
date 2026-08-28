#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=submit_stage4_Gd
#SBATCH --output=/scratch/elena/9Li/results/log/submit_stage4_Gd_%j.out
#SBATCH --error=/scratch/elena/9Li/results/log/submit_stage4_Gd_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00

echo "================================================================"
echo "Submitting Stage 4 for Gd (Refinement)"
echo "Time: $(date)"
echo "================================================================"

JOBS_DIR="/scratch/elena/9Li/jobs"
RESULTS_DIR="/scratch/elena/9Li/results"
GD_BASE_DIR="/scratch/elena/9Li/filtered_root/Gd"

# ------------------------------------------------------------------------------
# 1. Determine sample type
# ------------------------------------------------------------------------------

EXTRA_FLAGS="${EXTRA_ARGS}"
IS_BKG=0

if [[ "$EXTRA_FLAGS" == *"--bkg"* ]]; then
    IS_BKG=1
    echo ">> STAGE 4 LAUNCHER (Gd): BACKGROUND MODE DETECTED <<"
else
    echo ">> STAGE 4 LAUNCHER (Gd): SIGNAL MODE DETECTED <<"
fi

# ------------------------------------------------------------------------------
# 2. Build the list of Stage 4 tasks
#
# Stage 4 operates on the multilateration outputs produced by Stage 2.
# The task list is constructed deterministically from the existing files
# and sorted by (run, chunk).
# ------------------------------------------------------------------------------

echo ""
echo "================================================================"
echo "Building Stage 4 task list"
echo "================================================================"

TASK_MAP="${RESULTS_DIR}/Gd_stage4_task_map_$([[ $IS_BKG -eq 1 ]] && echo bkg || echo signal).pkl"

echo "Task map: ${TASK_MAP}"

python3 - "$RESULTS_DIR" "$GD_BASE_DIR" "$IS_BKG" "$TASK_MAP" <<'PY'
import os
import sys
import pickle

results_dir = sys.argv[1]
base_dir = sys.argv[2]
is_bkg = int(sys.argv[3])
output_map = sys.argv[4]

subdirs = ["p_270", "p_350"]

all_tasks = []

# --------------------------------------------------------------------------
# Find all Gd runs corresponding to the selected sample type.
# --------------------------------------------------------------------------

for sub in subdirs:
    d = os.path.join(base_dir, sub)

    if not os.path.isdir(d):
        continue

    suffix = "_bkg.root" if is_bkg else "_signal.root"

    for filename in os.listdir(d):

        if not (
            filename.startswith("WCTE_merged_production_R")
            and filename.endswith(suffix)
        ):
            continue

        prefix = "WCTE_merged_production_R"
        run_str = filename[len(prefix):-len(suffix)]

        try:
            run_num = int(run_str)
        except ValueError:
            continue

        run_proc_dir = os.path.join(
            results_dir,
            f"run{run_num}",
            "processed"
        )

        if not os.path.isdir(run_proc_dir):
            continue

        # --------------------------------------------------------------
        # Stage 2 multilateration output names
        # --------------------------------------------------------------

        if is_bkg:
            prefix_pkl = "Li9_clusters_chunk_"
            suffix_pkl = "_BKG_multilat.pkl"
        else:
            prefix_pkl = "Li9_clusters_chunk_"
            suffix_pkl = "_multilat.pkl"

        for pkl_name in os.listdir(run_proc_dir):

            if not pkl_name.startswith(prefix_pkl):
                continue

            if is_bkg:
                if not pkl_name.endswith(suffix_pkl):
                    continue
            else:
                if not pkl_name.endswith(suffix_pkl):
                    continue
                if "BKG" in pkl_name:
                    continue

            chunk_str = pkl_name[
                len(prefix_pkl):-len(suffix_pkl)
            ]

            try:
                chunk_num = int(chunk_str)
            except ValueError:
                continue

            pkl_path = os.path.join(run_proc_dir, pkl_name)

            all_tasks.append(
                {
                    "run": run_num,
                    "chunk_id": chunk_num,
                    "input_file": pkl_path,
                }
            )

# --------------------------------------------------------------------------
# Deterministic ordering
# --------------------------------------------------------------------------

all_tasks.sort(key=lambda x: (x["run"], x["chunk_id"]))

# --------------------------------------------------------------------------
# Save task map
# --------------------------------------------------------------------------

with open(output_map, "wb") as f:
    pickle.dump(all_tasks, f)

print(f"Created Stage 4 task map:")
print(f"  {output_map}")
print(f"Total Stage 4 tasks: {len(all_tasks)}")
PY

STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "ERROR: Failed to build Stage 4 task map."
    exit $STATUS
fi

# ------------------------------------------------------------------------------
# 3. Check task map
# ------------------------------------------------------------------------------

if [ ! -f "${TASK_MAP}" ]; then
    echo "ERROR: Stage 4 task map was not created:"
    echo "  ${TASK_MAP}"
    exit 1
fi

# ------------------------------------------------------------------------------
# 4. Get number of Stage 4 tasks
# ------------------------------------------------------------------------------

TOTAL_TASKS=$(python3 - "${TASK_MAP}" <<'PY'
import pickle
import sys

with open(sys.argv[1], "rb") as f:
    tasks = pickle.load(f)

print(len(tasks))
PY
)

STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "ERROR: Failed to read Stage 4 task map."
    exit $STATUS
fi

if [ -z "$TOTAL_TASKS" ] || [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "ERROR: No Stage 2 multilateration outputs found."
    echo "Stage 4 cannot be submitted."
    exit 1
fi

MAX_INDEX=$((TOTAL_TASKS - 1))
ARRAY_RANGE="0-${MAX_INDEX}%10"

echo ""
echo "================================================================"
echo "Stage 4 task map ready"
echo "================================================================"
echo "Sample type : $([[ $IS_BKG -eq 1 ]] && echo BKG || echo SIGNAL)"
echo "Task map    : ${TASK_MAP}"
echo "Total tasks : ${TOTAL_TASKS}"
echo "SLURM array : ${ARRAY_RANGE}"
echo "================================================================"

# ------------------------------------------------------------------------------
# 5. Submit Stage 4 refinement array
# ------------------------------------------------------------------------------

echo ""
echo "Submitting Stage 4 refinement array..."

JOB_OUT=$(sbatch \
    --array="${ARRAY_RANGE}" \
    --export=ALL,EXTRA_ARGS="${EXTRA_FLAGS}",TASK_MAP="${TASK_MAP}" \
    "${JOBS_DIR}/submit_refinement_Gd.sh")

STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "ERROR: Failed to submit Stage 4 Array for Gd!"
    exit $STATUS
fi

JOB_ID=$(echo "$JOB_OUT" | awk '{print $4}')

if [ -z "$JOB_ID" ]; then
    echo "ERROR: Failed to retrieve Stage 4 Job ID."
    exit 1
fi

echo "Stage 4 (Gd) submitted successfully."
echo "Stage 4 Job ID: ${JOB_ID}"

# ------------------------------------------------------------------------------
# 6. Submit Stage 5 launcher
#
# IMPORTANT:
# Stage 5 uses AFTEROK.
# Therefore Stage 5 starts ONLY if the complete Stage 4 array
# finishes successfully.
# ------------------------------------------------------------------------------

echo ""
echo "Submitting Stage 5 launcher..."
echo "Waiting for successful completion of Stage 4 Job ID: ${JOB_ID}"

JOB_OUT_5=$(sbatch \
    --dependency="afterok:${JOB_ID}" \
    --export=ALL,EXTRA_ARGS="${EXTRA_FLAGS}" \
    "${JOBS_DIR}/submit_stage5_Gd.sh")

STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "ERROR: Failed to submit Stage 5 launcher for Gd!"
    exit $STATUS
fi

JOB_ID_5=$(echo "$JOB_OUT_5" | awk '{print $4}')

if [ -z "$JOB_ID_5" ]; then
    echo "ERROR: Failed to retrieve Stage 5 Job ID."
    exit 1
fi

echo "Stage 5 launcher submitted successfully."
echo "Stage 5 Job ID: ${JOB_ID_5}"
echo "Dependency: afterok:${JOB_ID}"

# ------------------------------------------------------------------------------
# Final summary
# ------------------------------------------------------------------------------

echo ""
echo "================================================================"
echo "Stage 4 -> Stage 5 chain successfully submitted"
echo "================================================================"
echo "Sample type : $([[ $IS_BKG -eq 1 ]] && echo BKG || echo SIGNAL)"
echo "Stage 4     : ${JOB_ID}"
echo "Stage 5     : ${JOB_ID_5}"
echo ""
echo "Stage 5 will start ONLY if Stage 4 completes successfully."
echo "================================================================"

exit 0
