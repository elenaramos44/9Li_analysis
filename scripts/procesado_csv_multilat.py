#!/usr/bin/env python3
import pandas as pd
import ast
from glob import glob
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Convert multilateration CSVs to binary PKL files")
    parser.add_argument("--run", type=int, required=True, help="Run number to process")
    parser.add_argument("--outdir", type=str, required=True, help="Target destination directory 'processed'")
    return parser.parse_args()

def main():
    args = parse_args()
    run = args.run
    outdir = args.outdir

    # Input pattern path based on the dynamic run number
    input_pattern = f"/scratch/elena/9Li/results/run{run}/multilat_output/*_multilat_chi2.csv"
    input_files = glob(input_pattern)

    if not input_files:
        print(f"Warning: No CSV files were found matching {input_pattern}")
        return

    os.makedirs(outdir, exist_ok=True)
    print(f"--- Processing Run {run}: Found {len(input_files)} CSV files ---")

    for f in input_files:
        print(f"Converting {os.path.basename(f)}...")
        try:
            df = pd.read_csv(f)

            # Convert string representations back into functional lists (critical for proper PKL utility)
            for col in ['hit_times_ns', 'hit_slot_ids', 'hit_position_ids', 'hit_charges']:
                if col in df.columns:
                    # Safely use ast.literal_eval only if the element is an encoded string
                    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

            # Change extension from .csv to .pkl
            outname = os.path.basename(f).replace("_multilat_chi2.csv", "_BKG.pkl")
            df.to_pickle(os.path.join(outdir, outname))
        except Exception as e:
            print(f"Error processing file {f}: {e}")

    print(f"Conversion completely finished for Run {run}!\n")

if __name__ == "__main__":
    main()