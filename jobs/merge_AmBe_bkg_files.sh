#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name=merge_bkg_auto
#SBATCH --output=/scratch/elena/9Li/results/log/merge_bkg_%j.out
#SBATCH --error=/scratch/elena/9Li/results/log/merge_bkg_%j.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# Load ROOT environment
source /scratch/elena/root-6.26.04-install/bin/thisroot.sh

IN_DIR="/data/elena/data/AmBe/2384_bkg"
OUT_DIR="/scratch/elena/9Li/filtered_root/AmBe_bkg"

mkdir -p "$OUT_DIR"
mkdir -p "/scratch/elena/9Li/results/log"

OUT_FILE="${OUT_DIR}/WCTE_merged_production_R2384_bkg.root"

# Initial cleanup
rm -f "$OUT_FILE"

echo "================================================================"
echo "Merging TTree 'WCTEReadoutWindows' using TChain..."
echo "================================================================"

python3 - << EOF
import ROOT
import glob

input_files = sorted(glob.glob("${IN_DIR}/WCTE_offline_R2384S0P*.root"))
out_file = "${OUT_FILE}"

print(f"Files found: {len(input_files)}")

# Target the REAL tree: WCTEReadoutWindows
chain = ROOT.TChain("WCTEReadoutWindows")
for f in input_files:
    chain.Add(f)

total_entries = chain.GetEntries()
print(f"Total readout windows to merge: {total_entries}")

# Fast binary clone directly to disk without RAM issues
chain.Merge(out_file, "fast")
print(">> MERGE COMPLETED SUCCESSFULLY.")
EOF

echo "================================================================"
echo "Merged file available at: $OUT_FILE"
echo "================================================================"