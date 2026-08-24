#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=merge_NiCf
#SBATCH --output=/scratch/elena/9Li/results/log/merge_NiCf_%j.out
#SBATCH --error=/scratch/elena/9Li/results/log/merge_NiCf_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# ================================================================
# Merge NiCf ROOT files
#
# Usage:
#   sbatch merge_NiCf_files.sh 2437
#   sbatch merge_NiCf_files.sh 2482
#   sbatch merge_NiCf_files.sh 2494
#   sbatch merge_NiCf_files.sh 2504
#   sbatch merge_NiCf_files.sh 2507
#   sbatch merge_NiCf_files.sh 2508
#
# Input:
#   /data/elena/data/NiCf/<RUN_NUMBER>/
#
# Output:
#   /scratch/elena/9Li/filtered_root/NiCf_bkg/
#   WCTE_merged_production_R<RUN_NUMBER>_bkg.root
# ================================================================

# Load ROOT environment
source /scratch/elena/root-6.26.04-install/bin/thisroot.sh

# ------------------------------------------------
# Get run number
# ------------------------------------------------

if [ -z "$1" ]; then
    echo "ERROR: No run number provided."
    echo ""
    echo "Usage:"
    echo "  sbatch merge_NiCf_files.sh <RUN_NUMBER>"
    echo ""
    echo "Allowed runs:"
    echo "  2437 2482 2494 2504 2507 2508"
    exit 1
fi

RUN_NUMBER=$1

# ------------------------------------------------
# Check run number
# ------------------------------------------------

case "$RUN_NUMBER" in
    2437|2482|2494|2504|2507|2508)
        ;;
    *)
        echo "ERROR: Unsupported NiCf run: ${RUN_NUMBER}"
        echo "Allowed runs: 2437 2482 2494 2504 2507 2508"
        exit 1
        ;;
esac

# ------------------------------------------------
# Input / output
# ------------------------------------------------

IN_DIR="/data/elena/data/NiCf/${RUN_NUMBER}"

OUT_DIR="/scratch/elena/9Li/filtered_root/NiCf_bkg"

OUT_FILE="${OUT_DIR}/WCTE_merged_production_R${RUN_NUMBER}_bkg.root"

# ------------------------------------------------
# Create output directories
# ------------------------------------------------

mkdir -p "$OUT_DIR"
mkdir -p "/scratch/elena/9Li/results/log"

# ------------------------------------------------
# Check input directory
# ------------------------------------------------

if [ ! -d "$IN_DIR" ]; then
    echo "ERROR: Input directory does not exist:"
    echo "$IN_DIR"
    exit 1
fi

# ------------------------------------------------
# Find input files
# ------------------------------------------------

N_FILES=$(find "$IN_DIR" -maxdepth 1 \
    -type f \
    -name "WCTE_offline_R${RUN_NUMBER}S0P*.root" | wc -l)

if [ "$N_FILES" -eq 0 ]; then
    echo "ERROR: No ROOT files found for Run ${RUN_NUMBER}"
    echo "Expected files matching:"
    echo "${IN_DIR}/WCTE_offline_R${RUN_NUMBER}S0P*.root"
    exit 1
fi

# ------------------------------------------------
# Remove previous output
# ------------------------------------------------

rm -f "$OUT_FILE"

# ------------------------------------------------
# Print configuration
# ------------------------------------------------

echo "================================================================"
echo "NiCf ROOT file merging"
echo "================================================================"
echo "Run number:       ${RUN_NUMBER}"
echo "Input directory:  ${IN_DIR}"
echo "Number of files:  ${N_FILES}"
echo "Output directory: ${OUT_DIR}"
echo "Output file:      ${OUT_FILE}"
echo "================================================================"

# ------------------------------------------------
# Merge using TChain
# ------------------------------------------------

python3 - << EOF
import ROOT
import glob
import sys

input_files = sorted(
    glob.glob("${IN_DIR}/WCTE_offline_R${RUN_NUMBER}S0P*.root")
)

out_file = "${OUT_FILE}"

print(f"Files found: {len(input_files)}")

if len(input_files) == 0:
    print("ERROR: No input files found.")
    sys.exit(1)

chain = ROOT.TChain("WCTEReadoutWindows")

for f in input_files:
    print(f"Adding: {f}")
    chain.Add(f)

total_entries = chain.GetEntries()

print("---------------------------------------------------------------")
print(f"Total readout windows to merge: {total_entries}")
print("---------------------------------------------------------------")

if total_entries == 0:
    print("ERROR: TChain contains zero entries.")
    sys.exit(1)

result = chain.Merge(out_file, "fast")

if result <= 0:
    print("ERROR: ROOT TChain::Merge failed.")
    sys.exit(1)

print(">> MERGE COMPLETED SUCCESSFULLY.")
EOF

# ------------------------------------------------
# Final check
# ------------------------------------------------

if [ -f "$OUT_FILE" ]; then
    echo "================================================================"
    echo "MERGE SUCCESSFUL"
    echo "Merged file:"
    echo "$OUT_FILE"
    echo "================================================================"
else
    echo "ERROR: Output file was not created."
    exit 1
fi