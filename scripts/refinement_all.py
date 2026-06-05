import os
import re
import sys
import glob
import argparse
import pandas as pd
import numpy as np

# ----------------------------------------------------------------------
# ENVIRONMENT AND PATH SETUP
# ----------------------------------------------------------------------
ROOT_PATH = "/scratch/elena/root-6.26.04-install"
os.environ["ROOTSYS"] = ROOT_PATH
os.environ["PYTHONPATH"] = f"{ROOT_PATH}/lib:{os.environ.get('PYTHONPATH', '')}"
os.environ["LD_LIBRARY_PATH"] = f"{ROOT_PATH}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
sys.path.append(f"{ROOT_PATH}/lib")

os.environ["WCSIM_BUILD_DIR"] = "/scratch/elena/wcsim-install"
os.environ["BONSAIDIR"] = "/scratch/elena/bonsai"

sys.path.append("/scratch/elena/9Li/scripts")
import functions_bonsai
import functions_multilateration

# Constant speed of light in the medium (cm/ns)
C_N = 29.9792458 / 1.33  

# ----------------------------------------------------------------------
# REFINEMENT PIPELINE FUNCTION
# ----------------------------------------------------------------------
def refine_cluster(row, lookup_table):
    """Refines a single cluster vertex using multilateration after filtering hits."""
    # Force module namespace initialization
    functions_bonsai.geo = lookup_table
    
    times = np.array(row['hit_times_ns'])
    mpmt_ids = np.array(row['hit_slot_ids'])
    pmt_ids = np.array(row['hit_position_ids'])
    v_init = np.array([row['vertex_x'], row['vertex_y'], row['vertex_z']])
    
    try:
        # Get physical PMT coordinates
        x_p, y_p, z_p, _ = functions_bonsai.getxyz(lookup_table, mpmt_ids, pmt_ids)
        pmt_pos = np.column_stack([x_p, y_p, z_p])
        
        # Calculate residual times (dt)
        tof = np.linalg.norm(pmt_pos - v_init, axis=1) / C_N
        t_corr = times - tof
        t0_guess = np.median(t_corr)
        dt = t_corr - t0_guess
        
        # Filter residual times (dt < 3.0 ns)
        clean_mask = (np.abs(dt) < 3.0)
        nhits_fine = np.sum(clean_mask)
        
        # CRITICAL FILTER: Drop cluster completely if remaining hits < 15
        if nhits_fine < 15:
            return pd.Series([np.nan]*5, index=['v_x_fine', 'v_y_fine', 'v_z_fine', 't_rms_fine', 'nhits_fine'])

        # Multilateration fit (Fine adjustment)
        vertex = functions_multilateration.run_multilateration_candidate(
            times[clean_mask], mpmt_ids[clean_mask], pmt_ids[clean_mask],
            sigma_t=2.2,
            guess=(v_init[0], v_init[1], v_init[2], t0_guess)
        )
        
        if vertex["success"]:
            t_rms_final = np.std(vertex["pulls"] * 2.2)
            return pd.Series([vertex['x'], vertex['y'], vertex['z'], t_rms_final, nhits_fine], 
                             index=['v_x_fine', 'v_y_fine', 'v_z_fine', 't_rms_fine', 'nhits_fine'])
    except Exception:
        pass
    
    return pd.Series([np.nan]*5, index=['v_x_fine', 'v_y_fine', 'v_z_fine', 't_rms_fine', 'nhits_fine'])


# ----------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Parallel Refinement Processing for 9Li Data Chunks")
    parser.add_argument("--run", type=int, required=True, help="Run number to process")
    parser.add_argument("--chunk-id", type=int, required=True, help="Specific chunk ID to process")
    args = parser.parse_args()

    processed_folder = f"/scratch/elena/9Li/results/run{args.run}/processed"
    
    # Locate the unique input file containing the specified chunk ID
    search_pattern = f"{processed_folder}/*(15-50)*chunk_{args.chunk_id}.pkl"
    matching_files = glob.glob(search_pattern)
    
    if not matching_files:
        print(f"Error: No matching file found for Run {args.run}, Chunk {args.chunk_id}")
        sys.exit(1)
        
    input_filepath = matching_files[0]
    output_filename = f"Refine_Li9_clusters_chunk_{args.chunk_id}.pkl"
    output_filepath = os.path.join(processed_folder, output_filename)
    
    print(f"Loading Geometry Mapping...")
    geo_data = functions_bonsai.get_geo_mapping()
    lookup_table = functions_bonsai.build_lookup_table(geo_data)
    functions_bonsai.geo = lookup_table

    print(f"Processing input file: {os.path.basename(input_filepath)}")
    df_chunk = pd.read_pickle(input_filepath)
    
    if df_chunk.empty:
        print("Input dataframe is empty. Writing empty file to disk.")
        df_empty = pd.DataFrame(columns=list(df_chunk.columns) + ['v_x_fine', 'v_y_fine', 'v_z_fine', 't_rms_fine', 'nhits_fine'])
        df_empty.to_pickle(output_filepath)
        return

    # Preliminary filter: fit success, time_rms, and fiducial volume constraints
    mask_pre = (df_chunk['fit_success'] == True) & \
               (df_chunk['time_rms'] < 3.0) & \
               (df_chunk['vertex_x'].abs() < 270) & \
               (df_chunk['vertex_y'].abs() < 270)
               
    df_to_refine = df_chunk[mask_pre].copy()
    print(f"Candidates passing pre-filter: {len(df_to_refine)} / {len(df_chunk)}")

    if df_to_refine.empty:
        print("No candidates to refine. Writing empty results.")
        df_empty = pd.DataFrame(columns=list(df_chunk.columns) + ['v_x_fine', 'v_y_fine', 'v_z_fine', 't_rms_fine', 'nhits_fine'])
        df_empty.to_pickle(output_filepath)
        return

    # Apply pipeline to every cluster row
    refined_results = df_to_refine.apply(refine_cluster, axis=1, args=(lookup_table,))
    df_final = pd.concat([df_to_refine, refined_results], axis=1)
    
    # Drop records that were flagged with NaN due to failing < 15 hits condition or optimization limits
    df_final = df_final.dropna(subset=['t_rms_fine'])
    
    # Save output file
    df_final.to_pickle(output_filepath)
    print(f"Successfully saved refinement data to {output_filepath}")
    print(f"Final valid clusters remaining: {len(df_final)}")

if __name__ == "__main__":
    main()