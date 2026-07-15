#!/usr/bin/env python3
import pandas as pd
from glob import glob
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Reformat multilateration outputs into standard stage filenames.")
    parser.add_argument("--run", type=int, required=True, help="Run number to process")
    parser.add_argument("--outdir", type=str, required=True, help="Target destination directory 'processed'")
    parser.add_argument("--bkg", action="store_true", help="Process background files instead of signal")
    return parser.parse_args()

def main():
    args = parse_args()
    run = args.run
    outdir = args.outdir

    # Safely find chunks based on dataset target type
    if args.bkg:
        input_pattern = os.path.join(outdir, "*_BKG_multilat.pkl")
        input_files = glob(input_pattern)
        sample_label = "BACKGROUND"
    else:
        input_pattern = os.path.join(outdir, "*_multilat.pkl")
        # Explicitly exclude background files to avoid overlaps
        input_files = [f for f in glob(input_pattern) if "_BKG_" not in os.path.basename(f)]
        sample_label = "SIGNAL"

    if not input_files:
        print(f"Warning: No PKL files were found matching {input_pattern} for {sample_label}")
        return

    print(f"--- Stage 3 [{sample_label}] Run {run}: Standardizing {len(input_files)} Chunks ---")

    for f in input_files:
        try:
            df = pd.read_pickle(f)

            # Replaces the multilat tag back to a standard stage chunk file format
            outname = os.path.basename(f).replace("_multilat.pkl", ".pkl")
            output_filepath = os.path.join(outdir, outname)
            
            df.to_pickle(output_filepath)
            
        except Exception as e:
            print(f"Error standardizing file {f}: {e}")

    print(f"Stage 3 normalization completely finished for Run {run}!\n")

if __name__ == "__main__":
    main()