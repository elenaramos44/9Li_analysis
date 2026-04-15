#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import os
import sys

# Add path to your scripts
sys.path.append("/scratch/elena/9Li/scripts")
import functions_bonsai
import functions_multilateration

def parse_args():
    parser = argparse.ArgumentParser(description="Multilateration vertex reconstruction with Quality Metric (No discard)")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV with clusters")
    parser.add_argument("--outdir", type=str, required=True, help="Output folder")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

def run_multilat_full_info(row, verbose=False):
    """
    Runs multilateration and extracts Quality Metric (Time RMS).
    Cleans data but returns NaNs for failed fits to keep row consistency.
    """
    times = np.array(row['hit_times_ns'])
    mpmt_ids = np.array(row['hit_slot_ids'])
    pmt_ids  = np.array(row['hit_channel_ids'])

    # 1. CLEANING: Remove ghost hits (ID -1) and non-finite times
    valid_mask = (mpmt_ids >= 0) & (pmt_ids >= 0) & np.isfinite(times)
    times = times[valid_mask]
    mpmt_ids = mpmt_ids[valid_mask]
    pmt_ids = pmt_ids[valid_mask]

    # Default data structure for failed fits
    res_data = {
        "vertex_x": np.nan,
        "vertex_y": np.nan,
        "vertex_z": np.nan,
        "fit_success": False,
        "n_hits_used": len(times),
        "time_rms": np.nan
    }

    # Algorithm requirement: minimum 6 hits for a 4D fit (x,y,z,t)
    if len(times) < 6:
        return res_data

    try:
        # Run the robust fit
        # This will use functions_bonsai.geo which we initialize in main()
        vertex = functions_multilateration.run_multilateration_candidate(
            times, mpmt_ids, pmt_ids,
            sigma_t=1.0,
            early_window_ns=100.0,
            robust_loss="soft_l1"
        )

        if vertex["success"] and vertex["result"] is not None:
            # Calculate Time RMS from the cost function residuals
            # vertex["result"].fun returns (times - model) / sigma
            residuals = vertex["result"].fun
            time_rms = np.std(residuals)

            res_data.update({
                "vertex_x": vertex["x"],
                "vertex_y": vertex["y"],
                "vertex_z": vertex["z"],
                "fit_success": True,
                "n_hits_used": vertex["n_hits_used"],
                "time_rms": time_rms
            })

    except Exception as e:
        if verbose:
            print(f"Error in cluster: {e}")
        pass

    return res_data

def main():
    args = parse_args()

    # --- GEOMETRY INITIALIZATION ---
    # We must set functions_bonsai.geo so that functions_multilateration can find it
    if args.verbose:
        print("Initializing geometry and lookup tables...")
    
    geo_df = functions_bonsai.get_geo_mapping()
    # This is the line that fixes the 'attribute geo' error:
    functions_bonsai.geo = functions_bonsai.build_lookup_table(geo_df)

    # Load cluster data
    df = pd.read_csv(args.csv, converters={
        'hit_times_ns': eval, 'hit_slot_ids': eval, 'hit_channel_ids': eval
    })

    if args.verbose:
        print(f"Loaded {len(df)} clusters. Starting reconstruction...")

    results = []
    for i, row in df.iterrows():
        if args.verbose and i % 500 == 0:
            print(f"Processing cluster {i}/{len(df)}...")

        v_info = run_multilat_full_info(row, verbose=args.verbose)
        
        # Merge original cluster info (energy, time) with new vertex info
        combined_row = {**row, **v_info}
        results.append(combined_row)

    # Save everything to a single CSV
    df_final = pd.DataFrame(results)
    
    os.makedirs(args.outdir, exist_ok=True)
    out_name = os.path.basename(args.csv).replace(".csv", "_multilat_full.csv")
    out_path = os.path.join(args.outdir, out_name)
    
    df_final.to_csv(out_path, index=False)
    
    if args.verbose:
        print(f"Finished! Saved {len(df_final)} rows to: {out_path}")

if __name__ == "__main__":
    main()