import os
import glob
import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Merge refined chunks and apply strict Fiducial Volume selection without plotting.")
    parser.add_argument("--run", type=int, required=True, help="Run number to process")
    # Adds interactive flag: if included -> Background, if omitted -> Signal (default for sbatch)
    parser.add_argument("--bkg", action="store_true", help="Process background files instead of signal")
    args = parser.parse_args()

    run_dir = f"/scratch/elena/9Li/results/run{args.run}/processed"
    
    # Define search patterns and labels dynamically based on the --bkg flag
    if args.bkg:
        search_pattern = os.path.join(run_dir, "Refine_Li9_clusters_chunk*_BKG.pkl")
        sample_label = "BACKGROUND"
        output_filename = f"Final_FV_Li9_clusters_run{args.run}_BKG.pkl"
    else:
        search_pattern = os.path.join(run_dir, "Refine_Li9_clusters_chunk*.pkl")
        sample_label = "SIGNAL"
        output_filename = f"Final_FV_Li9_clusters_run{args.run}.pkl"

    refined_files = sorted(glob.glob(search_pattern))
    
    # Filter out background files if we are processing the Signal sample
    if not args.bkg:
        refined_files = [f for f in refined_files if not f.endswith('_BKG.pkl')]

    if not refined_files:
        print(f"Error: No refined chunk files found for Run {args.run} ({sample_label}) in {run_dir}")
        return

    print(f"Found {len(refined_files)} refined chunk files for Run {args.run} ({sample_label}). Merging...")
    
    # Concatenate all data chunks into a single DataFrame
    dfs = [pd.read_pickle(f) for f in refined_files]
    df_all_refined = pd.concat(dfs, ignore_index=True)
    print(f"Total refined clusters loaded: {len(df_all_refined)}")

    if df_all_refined.empty:
        print("Merged dataframe is empty. Terminating script.")
        return

    # --- TRACKING INITIAL STATS BEFORE FV CUT ---
    initial_clusters = len(df_all_refined)
    initial_spills = set(df_all_refined['spill_id'].unique()) if 'spill_id' in df_all_refined.columns else set()

    # FV limits
    x_lims = [-20.0, 20.0]
    y_lims = [-20.0, 20.0]    #y=0 in WCTE data!!
    z_lims = [-138.0, 0.0]

    fv_mask = (df_all_refined['v_x_fine'] >= x_lims[0]) & (df_all_refined['v_x_fine'] <= x_lims[1]) & \
              (df_all_refined['v_y_fine'] >= y_lims[0]) & (df_all_refined['v_y_fine'] <= y_lims[1]) & \
              (df_all_refined['v_z_fine'] >= z_lims[0]) & (df_all_refined['v_z_fine'] <= z_lims[1])

    df_final_fv = df_all_refined[fv_mask].copy()
    print(f"Clusters remaining inside strict FV: {len(df_final_fv)} / {len(df_all_refined)}")

    # --- TRACKING FINAL STATS AFTER FV CUT & PRINTING ANALYSIS ---
    final_clusters = len(df_final_fv)
    final_spills = set(df_final_fv['spill_id'].unique()) if 'spill_id' in df_final_fv.columns else set()

    # Calculate losses
    dropped_clusters = initial_clusters - final_clusters
    lost_spills = initial_spills - final_spills
    num_lost_spills = len(lost_spills)

    print("\n" + "="*65)
    print(f" STAGE ANALYSIS: FIDUCIAL VOLUME CUT FOR RUN {args.run} ({sample_label})")
    print("="*65)
    print(f"Before FV Cut : {initial_clusters} clusters across {len(initial_spills)} unique spills.")
    print(f"After FV Cut  : {final_clusters} clusters across {len(final_spills)} unique spills.")
    print("-"*65)
    print(f"DISCARDED BY FV CUT:")
    print(f"  -> Cluster candidates dropped   : {dropped_clusters} ({(dropped_clusters/initial_clusters)*100:.2f}% of total)")
    print(f"  -> Spills completely eliminated : {num_lost_spills}")
    
    if num_lost_spills > 0:
        # Sort for clean visualization in the log
        sorted_lost_spills = sorted(list(lost_spills))
        #print(f"  -> List of eliminated spill IDs : {sorted_lost_spills}")
    print("="*65 + "\n")

    output_pkl_path = os.path.join(run_dir, output_filename)
    df_final_fv.to_pickle(output_pkl_path)
    print(f"Successfully saved final selection structure to: {output_pkl_path}")

if __name__ == "__main__":
    main()