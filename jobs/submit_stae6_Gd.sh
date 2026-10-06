#!/bin/bash

# ============================================================================
# Stage 6 - Delayed neutron coincidence search
#
# Gd data ONLY
#
# Signal:
#   Final_FV_Li9_clusters_runXXX.pkl
#
# Background:
#   Final_FV_Li9_clusters_runXXX_BKG.pkl
#
# The same spill-aware chunk map used during the Li9 processing
# must be supplied through CHUNK_MAP.
# ============================================================================

set -u

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

SCRIPT="/scratch/elena/9Li/scripts/delayed_neutron_search.py"

# IMPORTANT:
# Set this to the Gd spill-aware chunk map used for the corresponding ROOT
# files (p_270 / p_350).
#
# Example:
# CHUNK_MAP="/scratch/elena/9Li/results/chunk_map_Gd.pkl"
#
# Or override when submitting:
#
# CHUNK_MAP=/path/to/map.pkl bash submit_stage6_Gd.sh
#
CHUNK_MAP="${CHUNK_MAP:-/scratch/elena/9Li/results/chunk_map_Gd.pkl}"

FVTAG="${FVTAG:-FV_1}"

GAP_US="${GAP_US:-5}"
SEARCH_US="${SEARCH_US:-150}"

PROMPT_WINDOW_NS="${PROMPT_WINDOW_NS:-20}"

WINDOW_NS="${WINDOW_NS:-10}"
NHITS_MIN="${NHITS_MIN:-10}"
NHITS_MAX="${NHITS_MAX:-50}"
RMS_CUT_NS="${RMS_CUT_NS:-10}"

OUTTAG="${OUTTAG:-}"

# Set to 1 if you want to submit the BKG sample too.
RUN_BACKGROUND="${RUN_BACKGROUND:-1}"

# ----------------------------------------------------------------------------
# Check configuration
# ----------------------------------------------------------------------------

echo "================================================================"
echo "Stage 6 - Gd delayed neutron coincidence search"
echo "================================================================"
echo "Script          : ${SCRIPT}"
echo "Chunk map       : ${CHUNK_MAP}"
echo "FV              : ${FVTAG}"
echo "Prompt window   : ${PROMPT_WINDOW_NS} ns"
echo "Gap             : ${GAP_US} us"
echo "Search          : ${SEARCH_US} us"
echo "Delayed window  : ${WINDOW_NS} ns"
echo "Delayed nHits   : ${NHITS_MIN} - ${NHITS_MAX}"
echo "Delayed RMS     : < ${RMS_CUT_NS} ns"
echo "Run BKG         : ${RUN_BACKGROUND}"
echo "================================================================"

if [ ! -f "${SCRIPT}" ]; then
    echo "ERROR: Python script not found:"
    echo "       ${SCRIPT}"
    exit 1
fi

if [ ! -f "${CHUNK_MAP}" ]; then
    echo "ERROR: Gd chunk map not found:"
    echo "       ${CHUNK_MAP}"
    echo
    echo "Set it with:"
    echo
    echo "  CHUNK_MAP=/path/to/Gd_chunk_map.pkl bash submit_stage6_Gd.sh"
    exit 1
fi

# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------

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


# ============================================================================
# Determine number of chunks
# ============================================================================

NCHUNKS=$(python3 - "${CHUNK_MAP}" <<'PY'
import sys
import pickle

path = sys.argv[1]

with open(path, "rb") as f:
    chunks = pickle.load(f)

if not chunks:
    raise RuntimeError("Chunk map is empty.")

ids = [
    int(c["chunk_id"])
    for c in chunks
]

print(max(ids) + 1)
PY
)

if [ -z "${NCHUNKS}" ] || [ "${NCHUNKS}" -le 0 ]; then
    echo "ERROR: Could not determine number of chunks."
    exit 1
fi

LAST_CHUNK=$((NCHUNKS - 1))

echo
echo "Number of chunks : ${NCHUNKS}"
echo "Array            : 0-${LAST_CHUNK}"
echo


# ============================================================================
# Temporary SLURM job script
# ============================================================================

JOB_SCRIPT=$(mktemp)

cat > "${JOB_SCRIPT}" <<EOF
#!/bin/bash

#SBATCH --qos=regular
#SBATCH --job-name=Li9_S6_Gd
#SBATCH --output=/scratch/elena/9Li/results/log/stage6_Gd_%A_%a.out
#SBATCH --error=/scratch/elena/9Li/results/log/stage6_Gd_%A_%a.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=4:00:00

set -u

echo "================================================================"
echo "Starting Stage 6 - Gd delayed neutron search"
echo "Time      : \$(date)"
echo "Job ID    : \${SLURM_JOB_ID}"
echo "Array ID  : \${SLURM_ARRAY_TASK_ID}"
echo "================================================================"

source /scicomp/builds/Rocky/8.7/Common/software/Miniforge3/24.11.3-2/etc/profile.d/conda.sh

conda activate /scratch/elena/conda-env/wcsim-env

source /scratch/elena/root-6.26.04-install/bin/thisroot.sh

source /scratch/elena/geant4.10.03.p03-install/bin/geant4.sh

export Geant4_DIR=/scratch/elena/geant4.10.03.p03-install/lib64/Geant4-10.3.3/Geant4Config.cmake

export WCSIM_BUILD_DIR=/scratch/elena/wcsim-install

export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/scratch/elena/wcsim-install/lib

export BONSAIDIR=/scratch/elena/bonsai

export LD_LIBRARY_PATH=\$BONSAIDIR:\$LD_LIBRARY_PATH

export ROOT_INCLUDE_PATH=\$BONSAIDIR/bonsai:/scratch/elena/wcsim-install/include/WCSim:\$ROOT_INCLUDE_PATH


python3 "${SCRIPT}" \\
    --chunk-map "${CHUNK_MAP}" \\
    --chunk-id "\${SLURM_ARRAY_TASK_ID}" \\
    --fvtag "${FVTAG}" \\
    --prompt-window-ns "${PROMPT_WINDOW_NS}" \\
    --gap-us "${GAP_US}" \\
    --search-us "${SEARCH_US}" \\
    --window-ns "${WINDOW_NS}" \\
    --nhits-min "${NHITS_MIN}" \\
    --nhits-max "${NHITS_MAX}" \\
    --rms-cut-ns "${RMS_CUT_NS}" \\
    --outtag "${OUTTAG}" \\
    --verbose

STATUS=\$?

if [ \$STATUS -ne 0 ]; then
    echo "ERROR: Stage 6 SIGNAL failed."
    exit \$STATUS
fi

echo "Stage 6 SIGNAL completed successfully."
echo "Time: \$(date)"

EOF


# ============================================================================
# Submit SIGNAL
# ============================================================================

echo "Submitting SIGNAL array..."

SIGNAL_JOB=$(sbatch \
    --array=0-${LAST_CHUNK}%10 \
    "${JOB_SCRIPT}" \
    | awk '{print $4}')

echo
echo "Submitted SIGNAL:"
echo "  Job ID: ${SIGNAL_JOB}"
echo


# ============================================================================
# Submit BACKGROUND
# ============================================================================

if [ "${RUN_BACKGROUND}" -eq 1 ]; then

    BKG_JOB_SCRIPT=$(mktemp)

    sed 's/--verbose/--bkg \\\n    --verbose/' \
        "${JOB_SCRIPT}" \
        > "${BKG_JOB_SCRIPT}"

    echo "Submitting BACKGROUND array..."

    BKG_JOB=$(sbatch \
        --array=0-${LAST_CHUNK}%10 \
        --job-name=Li9_S6_Gd_BKG \
        --output=/scratch/elena/9Li/results/log/stage6_Gd_BKG_%A_%a.out \
        --error=/scratch/elena/9Li/results/log/stage6_Gd_BKG_%A_%a.err \
        "${BKG_JOB_SCRIPT}" \
        | awk '{print $4}')

    echo
    echo "Submitted BACKGROUND:"
    echo "  Job ID: ${BKG_JOB}"
    echo

    rm -f "${BKG_JOB_SCRIPT}"

fi


# ----------------------------------------------------------------------------
# Clean temporary script
# ----------------------------------------------------------------------------

rm -f "${JOB_SCRIPT}"

echo "================================================================"
echo "Stage 6 Gd submission completed."
echo "================================================================"