#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=merge_AmBe
#SBATCH --output=/scratch/elena/9Li/results/log/merge_AmBe_%j.out
#SBATCH --error=/scratch/elena/9Li/results/log/merge_AmBe_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# ================================================================
# Merge AmBe ROOT files
#
# Usage:
#   sbatch merge_AmBe_files.sh 2384
#   sbatch merge_AmBe_files.sh 2387
#   sbatch merge_AmBe_files.sh 2388
#   sbatch merge_AmBe_files.sh 2389
#   sbatch merge_AmBe_files.sh 2390
#
# Run 2384:
#   Input:  /data/elena/data/AmBe/2384_bkg
#   Output: /scratch/elena/9Li/filtered_root/AmBe_bkg/
#           WCTE_merged_production_R2384_bkg.root
#
# Runs 2387-2390:
#   Input:  /data/elena/data/AmBe/signal/<RUN>_sig
#   Output: /scratch/elena/9Li/filtered_root/AmBe_sig/
#           WCTE_merged_production_R<RUN>_signal.root
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
    echo "  sbatch merge_AmBe_files.sh <RUN_NUMBER>"
    echo ""
    echo "Examples:"
    echo "  sbatch merge_AmBe_files.sh 2384"
    echo "  sbatch merge_AmBe_files.sh 2387"
    echo "  sbatch merge_AmBe_files.sh 2388"
    echo "  sbatch merge_AmBe_files.sh 2389"
    echo "  sbatch merge_AmBe_files.sh 2390"
    exit 1
fi

RUN_NUMBER=$1

# ------------------------------------------------
# Determine input/output directories and filename
# ------------------------------------------------

if [ "$RUN_NUMBER" -eq 2384 ]; then

    # Original AmBe background run
    IN_DIR="/data/elena/data/AmBe/2384_bkg"
    OUT_DIR="/scratch/elena/9Li/filtered_root/AmBe_bkg"
    OUT_FILE="${OUT_DIR}/WCTE_merged_production_R${RUN_NUMBER}_bkg.root"

else

    # AmBe signal runs
    IN_DIR="/data/elena/data/AmBe/signal/${RUN_NUMBER}_sig"
    OUT_DIR="/scratch/elena/9Li/filtered_root/AmBe_sig"
    OUT_FILE="${OUT_DIR}/WCTE_merged_production_R${RUN_NUMBER}_signal.root"

fi

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
echo "AmBe ROOT file merging"
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

# Target the real tree
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

# Fast binary clone directly to disk without loading everything into RAM
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