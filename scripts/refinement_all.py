#!/usr/bin/env python3
import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd

# Setup ROOT & WCTE environment
ROOT_PATH = "/scratch/elena/root-6.26.04-install"
os.environ["ROOTSYS"] = ROOT_PATH
os.environ["PYTHONPATH"] = f"{ROOT_PATH}/lib:{os.environ.get('PYTHONPATH', '')}"
os.environ["LD_LIBRARY_PATH"] = (
    f"{ROOT_PATH}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
)
sys.path.append(f"{ROOT_PATH}/lib")

os.environ["WCSIM_BUILD_DIR"] = "/scratch/elena/wcsim-install"
os.environ["BONSAIDIR"] = "/scratch/elena/bonsai"

sys.path.append("/scratch/elena/9Li/scripts")
import functions_multilateration
import geometry_wcte

# Speed of light in water for WCTE (cm/ns)
c_n = 29.9792458 / 1.333


def refine_cluster(row):
  """Applies time-residual cleaning and re-fits the vertex using multilateration."""
  # --- CUT 1: Reject if initial T_RMS > 3.0 ns ---
  if row.get("time_rms", np.inf) > 3.0:
    return pd.Series(
        [np.nan] * 5,
        index=[
            "v_x_fine",
            "v_y_fine",
            "v_z_fine",
            "t_rms_fine",
            "hits_after",
        ],
    )

  times = np.asarray(row["hit_times_ns"])
  mpmt_ids = np.asarray(row["hit_slot_ids"])
  pmt_ids = np.asarray(row["hit_position_ids"])

  # Initial seed vertex from stage 1
  vertex_seed = np.array(
      [row["vertex_x"], row["vertex_y"], row["vertex_z"]], dtype=float
  )

  try:
    # Geometry lookup
    x_p, y_p, z_p = geometry_wcte.get_xyz(mpmt_ids, pmt_ids, units="cm")
    pmt_pos = np.column_stack([x_p, y_p, z_p])

    # Time-residual calculation (dt) relative to initial seed
    tof = np.linalg.norm(pmt_pos - vertex_seed, axis=1) / c_n
    t_corr = times - tof
    t0_guess = np.median(t_corr)
    dt = t_corr - t0_guess

    # --- CUT 2: Reject individual hits with |dt| >= 3.0 ns ---
    clean_mask = np.abs(dt) < 3.0
    nhits_clean = np.sum(clean_mask)

    # --- CUT 3: Multiplicity requirement: 15 < clean_hits < 50 ---
    if not (15 < nhits_clean < 50):
      return pd.Series(
          [np.nan] * 5,
          index=[
              "v_x_fine",
              "v_y_fine",
              "v_z_fine",
              "t_rms_fine",
              "hits_after",
          ],
      )

    # Re-run multilateration with clean hits
    vertex = functions_multilateration.run_multilateration_candidate(
        times[clean_mask],
        mpmt_ids[clean_mask],
        pmt_ids[clean_mask],
        sigma_t=1.0,
        initial_vertex=vertex_seed,
    )

    if not vertex["success"]:
      return pd.Series(
          [np.nan] * 5,
          index=[
              "v_x_fine",
              "v_y_fine",
              "v_z_fine",
              "t_rms_fine",
              "hits_after",
          ],
      )

    residuals_fine = vertex["pulls"]
    t_rms_final = np.std(residuals_fine)
    hits_after = vertex["n_hits_used"]

    return pd.Series(
        [vertex["x"], vertex["y"], vertex["z"], t_rms_final, hits_after],
        index=["v_x_fine", "v_y_fine", "v_z_fine", "t_rms_fine", "hits_after"],
    )

  except Exception:
    return pd.Series(
        [np.nan] * 5,
        index=[
            "v_x_fine",
            "v_y_fine",
            "v_z_fine",
            "t_rms_fine",
            "hits_after",
        ],
    )


def main():
  parser = argparse.ArgumentParser(
      description="Strict refinement for 9Li candidates."
  )
  parser.add_argument("--run", type=int, required=True, help="run_number")
  parser.add_argument("--chunk-id", type=int, required=True, help="chunk_id")
  parser.add_argument(
      "--bkg", action="store_true", help="Process background sample"
  )
  args = parser.parse_args()

  processed_folder = f"/scratch/elena/9Li/results/run{args.run}/processed"

  # SIGNAL: clean name | BACKGROUND: tagged with _BKG
  if args.bkg:
    search_pattern = f"{processed_folder}/Li9_clusters_chunk_{args.chunk_id}_BKG_multilat.pkl"
    output_filename = f"Refine_Li9_clusters_chunk_{args.chunk_id}_BKG.pkl"
    sample_label = "BACKGROUND"
  else:
    search_pattern = f"{processed_folder}/Li9_clusters_chunk_{args.chunk_id}_multilat.pkl"
    output_filename = f"Refine_Li9_clusters_chunk_{args.chunk_id}.pkl"
    sample_label = "SIGNAL"

  matching_files = glob.glob(search_pattern)

  if not matching_files:
    print(
        f"Error: No file found for {sample_label} Run {args.run}, Chunk"
        f" {args.chunk_id}"
    )
    sys.exit(1)

  input_filepath = matching_files[0]
  output_filepath = os.path.join(processed_folder, output_filename)

  df_chunk = pd.read_pickle(input_filepath)
  columnas_nuevas = [
      "v_x_fine",
      "v_y_fine",
      "v_z_fine",
      "t_rms_fine",
      "hits_after",
  ]

  if df_chunk.empty:
    df_empty = pd.DataFrame(columns=list(df_chunk.columns) + columnas_nuevas)
    df_empty.to_pickle(output_filepath)
    return

  mask_pre = df_chunk["fit_success"] == True
  df_to_refine = df_chunk[mask_pre].copy()

  if df_to_refine.empty:
    df_empty = pd.DataFrame(columns=list(df_chunk.columns) + columnas_nuevas)
    df_empty.to_pickle(output_filepath)
    return

  refined_results = df_to_refine.apply(refine_cluster, axis=1)
  df_final = pd.concat([df_to_refine, refined_results], axis=1)

  df_final = df_final.dropna(subset=["t_rms_fine"])

  if not df_final.empty:
    df_final["hits_after"] = df_final["hits_after"].astype(int)

  print(f"\n[{sample_label} - Run {args.run} - Chunk {args.chunk_id}]")
  print(f"  Candidates with initial fit : {len(df_to_refine)}")
  print(f"  Survived strict refinement  : {len(df_final)}")
  if len(df_to_refine) > 0:
    print(
        "  Stage efficiency            :"
        f" {100*len(df_final)/len(df_to_refine):.2f}%"
    )

  df_final.to_pickle(output_filepath)


if __name__ == "__main__":
  main()