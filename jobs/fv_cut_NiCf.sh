#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=NiCf_final_fv
#SBATCH --output=/scratch/elena/9Li/results/log/merge_NiCf_task_%j.out
#SBATCH --error=/scratch/elena/9Li/results/log/merge_NiCf_task_%j.err
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

SCRIPT=/scratch/elena/9Li/scripts/merge_and_fv_cut.py

# ------------------------------------------------
# Get run number
# ------------------------------------------------

if [ -z "$1" ]; then
    echo "ERROR: No run number provided."
    echo "Usage: sbatch fv_cut_NiCf.sh <RUN_NUMBER>"
    exit 1
fi

TARGET_RUN=$1

# ------------------------------------------------
# Check run number
# ------------------------------------------------

case "$TARGET_RUN" in
    2437|2482|2494|2504|2507|2508)
        ;;
    *)
        echo "ERROR: Unsupported NiCf run: ${TARGET_RUN}"
        echo "Allowed runs: 2437 2482 2494 2504 2507 2508"
        exit 1
        ;;
esac

# ------------------------------------------------
# NiCf is always BACKGROUND
# ------------------------------------------------

BKG_FLAG="--bkg"
SAMPLE_TYPE="BACKGROUND"

# ------------------------------------------------
# Check input merged ROOT file
# ------------------------------------------------

INPUT_FILE="/scratch/elena/9Li/filtered_root/NiCf_bkg/WCTE_merged_production_R${TARGET_RUN}_bkg.root"

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input ROOT file does not exist:"
    echo "$INPUT_FILE"
    exit 1
fi

# ------------------------------------------------
# Run merge + FV selection
# ------------------------------------------------

echo "==============================================================="
echo "Starting Merge and FV Cut Selection [NiCf]"
echo "==============================================================="
echo "Run:         ${TARGET_RUN}"
echo "Sample type: ${SAMPLE_TYPE}"
echo "Input file:  ${INPUT_FILE}"
echo "Time:        $(date)"
echo "==============================================================="

python3 "$SCRIPT" \
    --run "$TARGET_RUN" \
    $BKG_FLAG

if [ $? -ne 0 ]; then
    echo "ERROR: merge_and_fv_cut.py failed."
    exit 1
fi

echo "==============================================================="
echo "Process completed successfully."
echo "Run:         ${TARGET_RUN}"
echo "Sample type: ${SAMPLE_TYPE}"
echo "==============================================================="