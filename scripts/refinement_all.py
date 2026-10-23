#!/usr/bin/env python3

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


# ==============================================================================
# Setup ROOT & WCTE environment
# ==============================================================================

ROOT_PATH = "/scratch/elena/root-6.26.04-install"

os.environ["ROOTSYS"] = ROOT_PATH
os.environ["PYTHONPATH"] = (
    f"{ROOT_PATH}/lib:{os.environ.get('PYTHONPATH', '')}"
)
os.environ["LD_LIBRARY_PATH"] = (
    f"{ROOT_PATH}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
)

sys.path.append(f"{ROOT_PATH}/lib")

os.environ["WCSIM_BUILD_DIR"] = "/scratch/elena/wcsim-install"
os.environ["BONSAIDIR"] = "/scratch/elena/bonsai"

sys.path.append("/scratch/elena/9Li/scripts")

import functions_multilaterator_wcte_version as functions_multilateration
import geometry_wcte


# ==============================================================================
# Configuration
# ==============================================================================

# Speed of light in water for WCTE (cm/ns).
# Must be consistent with the TimeCal multilaterator.
N_REFRACTION = 1.373
c_n = 29.9792458 / N_REFRACTION

# Existing analysis cuts.
MAX_INITIAL_RMS_NS = 3.0
MAX_HIT_RESIDUAL_NS = 3.0

MIN_CLEANED_HITS_EXCLUSIVE = 15
MAX_CLEANED_HITS_EXCLUSIVE = 50


# ==============================================================================
# Stage 3 refinement
# ==============================================================================

def refine_cluster(row):
    """
    Refine a candidate using time-residual cleaning and the TimeCal
    multilaterator.

    IMPORTANT:
      - The initial time_rms < 3 ns cut is NOT applied.
      - Its outcome is recorded in initial_rms_pass_3ns.
      - The individual-hit requirement abs(dt) < 3 ns is retained.
      - The cleaned-hit multiplicity requirement
        15 < nhits_clean < 50 is retained.
      - All candidates receive an output record, including failed fits.

    Returns
    -------
    pandas.Series
        Refined reconstruction, diagnostics and refinement status.
    """

    # --------------------------------------------------------------------------
    # Initialize the output record
    # --------------------------------------------------------------------------

    output = {
        "v_x_fine": np.nan,
        "v_y_fine": np.nan,
        "v_z_fine": np.nan,
        "t_rms_fine": np.nan,
        "hits_after": np.nan,
        "nhits_clean": np.nan,
        "refinement_success": False,
        "refinement_status": "not_processed",
        "chi2_fine": np.nan,
        "ndof_fine": np.nan,
        "chi2_ndof_fine": np.nan,
        "p_value_fine": np.nan,
        "initial_rms_pass_3ns": False,
    }

    # --------------------------------------------------------------------------
    # Record the initial RMS-cut decision without applying the cut
    # --------------------------------------------------------------------------

    try:
        initial_rms = float(row.get("time_rms", np.nan))
    except (TypeError, ValueError):
        initial_rms = np.nan

    output["initial_rms_pass_3ns"] = bool(
        np.isfinite(initial_rms)
        and initial_rms <= MAX_INITIAL_RMS_NS
    )

    # --------------------------------------------------------------------------
    # Validate the initial reconstructed vertex
    # --------------------------------------------------------------------------

    try:
        vertex_seed = np.array(
            [
                row["vertex_x"],
                row["vertex_y"],
                row["vertex_z"],
            ],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError):
        output["refinement_status"] = "invalid_initial_vertex"
        return pd.Series(output)

    if not np.isfinite(vertex_seed).all():
        output["refinement_status"] = "invalid_initial_vertex"
        return pd.Series(output)

    # --------------------------------------------------------------------------
    # Read and validate the original cluster hits
    # --------------------------------------------------------------------------

    try:
        times = np.asarray(row["hit_times_ns"], dtype=float)
        mpmt_ids = np.asarray(row["hit_slot_ids"])
        pmt_ids = np.asarray(row["hit_position_ids"])
    except (KeyError, TypeError, ValueError):
        output["refinement_status"] = "invalid_hit_arrays"
        return pd.Series(output)

    if not (
        len(times) == len(mpmt_ids) == len(pmt_ids)
    ):
        output["refinement_status"] = "inconsistent_hit_arrays"
        return pd.Series(output)

    # Remove invalid times and negative identifiers.
    valid_mask = (
        np.isfinite(times)
        & (mpmt_ids >= 0)
        & (pmt_ids >= 0)
    )

    times = times[valid_mask]
    mpmt_ids = mpmt_ids[valid_mask]
    pmt_ids = pmt_ids[valid_mask]

    if len(times) == 0:
        output["nhits_clean"] = 0
        output["refinement_status"] = "no_valid_hits"
        return pd.Series(output)

    # --------------------------------------------------------------------------
    # Calculate time residuals relative to the initial vertex
    # --------------------------------------------------------------------------

    try:
        x_p, y_p, z_p = geometry_wcte.get_xyz(
            mpmt_ids,
            pmt_ids,
            units="cm",
        )

        pmt_pos = np.column_stack([x_p, y_p, z_p])

    except Exception as exc:
        output["refinement_status"] = (
            f"geometry_error:{type(exc).__name__}"
        )
        return pd.Series(output)

    # Remove hits with invalid geometry.
    valid_geometry = np.isfinite(pmt_pos).all(axis=1)

    times = times[valid_geometry]
    mpmt_ids = mpmt_ids[valid_geometry]
    pmt_ids = pmt_ids[valid_geometry]
    pmt_pos = pmt_pos[valid_geometry]

    if len(times) == 0:
        output["nhits_clean"] = 0
        output["refinement_status"] = "no_valid_geometry"
        return pd.Series(output)

    # Time of flight from the initial vertex to each PMT.
    tof = (
        np.linalg.norm(pmt_pos - vertex_seed, axis=1)
        / c_n
    )

    # Corrected hit times and residuals relative to their median.
    t_corr = times - tof
    t0_guess = np.median(t_corr)
    dt = t_corr - t0_guess

    # --------------------------------------------------------------------------
    # Existing individual-hit residual cut
    # --------------------------------------------------------------------------

    clean_mask = np.abs(dt) < MAX_HIT_RESIDUAL_NS

    nhits_clean = int(np.sum(clean_mask))
    output["nhits_clean"] = nhits_clean

    # --------------------------------------------------------------------------
    # Existing cleaned-hit multiplicity requirement
    # --------------------------------------------------------------------------

    if not (
        MIN_CLEANED_HITS_EXCLUSIVE
        < nhits_clean
        < MAX_CLEANED_HITS_EXCLUSIVE
    ):
        output["refinement_status"] = (
            "failed_cleaned_hit_multiplicity"
        )
        return pd.Series(output)

    # --------------------------------------------------------------------------
    # Refit the vertex using all valid cleaned hits
    # --------------------------------------------------------------------------

    try:
        vertex = functions_multilateration.run_multilateration_candidate(
            times[clean_mask],
            mpmt_ids[clean_mask],
            pmt_ids[clean_mask],
            robust_loss="huber",
            initial_vertex=vertex_seed,
            use_peak_window=False,
        )

    except Exception as exc:
        output["refinement_status"] = (
            f"refinement_exception:{type(exc).__name__}"
        )
        return pd.Series(output)

    if not vertex["success"]:
        output["refinement_status"] = "refined_fit_failed"
        return pd.Series(output)

    # --------------------------------------------------------------------------
    # Calculate the refined time RMS in ns
    # --------------------------------------------------------------------------

    residuals_fine_ns = np.asarray(
        vertex["residuals_ns"],
        dtype=float,
    )

    if (
        len(residuals_fine_ns) == 0
        or not np.isfinite(residuals_fine_ns).all()
    ):
        output["refinement_status"] = (
            "invalid_refined_time_residuals"
        )
        return pd.Series(output)

    t_rms_final = float(np.std(residuals_fine_ns))

    # --------------------------------------------------------------------------
    # Store the successful refined reconstruction
    # --------------------------------------------------------------------------

    output.update({
        "v_x_fine": vertex["x"],
        "v_y_fine": vertex["y"],
        "v_z_fine": vertex["z"],
        "t_rms_fine": t_rms_final,
        "hits_after": int(vertex["n_hits_used"]),
        "refinement_success": True,
        "refinement_status": "success",
        "chi2_fine": vertex["chi2"],
        "ndof_fine": vertex["ndof"],
        "chi2_ndof_fine": vertex["chi2_ndof"],
        "p_value_fine": vertex["p_value"],
    })

    return pd.Series(output)


# ==============================================================================
# Main
# ==============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Stage 3 refinement for 9Li candidates using the "
            "TimeCal multilaterator. The initial RMS cut is recorded "
            "but not applied."
        )
    )

    parser.add_argument(
        "--run",
        type=int,
        required=True,
        help="Run number",
    )

    parser.add_argument(
        "--chunk-id",
        type=int,
        required=True,
        help="Chunk ID",
    )

    parser.add_argument(
        "--bkg",
        action="store_true",
        help="Process background sample",
    )

    args = parser.parse_args()

    processed_folder = (
        f"/scratch/elena/9Li/results/run{args.run}/processed"
    )

    # --------------------------------------------------------------------------
    # Input and output filenames
    # --------------------------------------------------------------------------

    if args.bkg:
        search_pattern = (
            f"{processed_folder}/"
            f"Li9_clusters_chunk_{args.chunk_id}_BKG_multilat.pkl"
        )

        output_filename = (
            f"Refine_Li9_clusters_chunk_{args.chunk_id}_BKG.pkl"
        )

        sample_label = "BACKGROUND"

    else:
        search_pattern = (
            f"{processed_folder}/"
            f"Li9_clusters_chunk_{args.chunk_id}_multilat.pkl"
        )

        output_filename = (
            f"Refine_Li9_clusters_chunk_{args.chunk_id}.pkl"
        )

        sample_label = "SIGNAL"

    matching_files = glob.glob(search_pattern)

    if not matching_files:
        print(
            f"Error: No file found for {sample_label}, "
            f"run {args.run}, chunk {args.chunk_id}."
        )
        sys.exit(1)

    input_filepath = matching_files[0]

    output_filepath = os.path.join(
        processed_folder,
        output_filename,
    )

    # --------------------------------------------------------------------------
    # Load Stage 2 output
    # --------------------------------------------------------------------------

    df_chunk = pd.read_pickle(input_filepath)

    print(
        f"\n[{sample_label} - Run {args.run} "
        f"- Chunk {args.chunk_id}]"
    )

    print(f"Input file: {input_filepath}")
    print(f"Input candidates: {len(df_chunk)}")

    # --------------------------------------------------------------------------
    # Expected Stage 2 columns
    # --------------------------------------------------------------------------

    stage2_columns = [
        "spill_id",
        "hit_times_ns",
        "hit_slot_ids",
        "hit_position_ids",
        "vertex_x",
        "vertex_y",
        "vertex_z",
        "fit_success",
        "n_hits_used",
        "time_rms",
        "chi2",
        "ndof",
        "chi2_ndof",
    ]

    # --------------------------------------------------------------------------
    # Columns produced by this refinement stage
    # --------------------------------------------------------------------------

    refinement_columns = [
        "v_x_fine",
        "v_y_fine",
        "v_z_fine",
        "t_rms_fine",
        "hits_after",
        "nhits_clean",
        "refinement_success",
        "refinement_status",
        "chi2_fine",
        "ndof_fine",
        "chi2_ndof_fine",
        "p_value_fine",
        "initial_rms_pass_3ns",
    ]

    # Ensure the output schema is available, including for empty chunks.
    for column in stage2_columns:
        if column not in df_chunk.columns:
            df_chunk[column] = np.nan

    for column in refinement_columns:
        if column not in df_chunk.columns:
            df_chunk[column] = np.nan

    # --------------------------------------------------------------------------
    # Handle empty chunks
    # --------------------------------------------------------------------------

    if df_chunk.empty:
        df_chunk.to_pickle(output_filepath)

        print("Input chunk is empty; saved an empty output with the expected columns.")
        print(f"Output file: {output_filepath}")
        return

    # --------------------------------------------------------------------------
    # Identify successful Stage 2 fits
    # --------------------------------------------------------------------------

    stage2_success = (
        df_chunk["fit_success"]
        .fillna(False)
        .astype(bool)
    )

    # Candidates without successful Stage 2 fits are retained in the output.
    df_chunk.loc[
        ~stage2_success,
        "refinement_status",
    ] = "stage2_fit_failed"

    df_chunk.loc[
        ~stage2_success,
        "refinement_success",
    ] = False

    # --------------------------------------------------------------------------
    # Refine candidates with successful Stage 2 fits
    # --------------------------------------------------------------------------

    df_to_refine = df_chunk.loc[stage2_success].copy()

    if not df_to_refine.empty:

        refined_results = df_to_refine.apply(
            refine_cluster,
            axis=1,
        )

        # Assign results back to the original rows.
        # Failed refinements remain in the DataFrame.
        for column in refinement_columns:
            df_chunk.loc[
                df_to_refine.index,
                column,
            ] = refined_results[column]

    # --------------------------------------------------------------------------
    # Do NOT remove candidates with failed refinements
    # --------------------------------------------------------------------------

    # Deliberately do not use:
    #
    # df_final = df_final.dropna(subset=["t_rms_fine"])
    #
    # Every candidate from the Stage 2 input is retained.
    #
    # The initial time_rms < 3 ns cut is not applied here.
    #
    # hits_after remains NaN for failed fits, so do not cast the entire
    # column to int.

    # --------------------------------------------------------------------------
    # Print diagnostic summary
    # --------------------------------------------------------------------------

    n_input = len(df_chunk)

    n_stage2_success = int(stage2_success.sum())

    n_refinement_success = int(
        df_chunk["refinement_success"]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    n_initial_rms_pass = int(
        df_chunk["initial_rms_pass_3ns"]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    print(f"  Input candidates             : {n_input}")
    print(f"  Successful Stage 2 fits      : {n_stage2_success}")
    print(f"  Initial RMS <= 3 ns           : {n_initial_rms_pass}")
    print(f"  Successful refined fits      : {n_refinement_success}")
    print(f"  Saved candidates             : {len(df_chunk)}")

    if n_stage2_success > 0:
        print(
            "  Refinement success fraction : "
            f"{100.0 * n_refinement_success / n_stage2_success:.2f}%"
        )

    print("\nRefinement status:")
    print(
        df_chunk["refinement_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    # --------------------------------------------------------------------------
    # Save all candidates
    # --------------------------------------------------------------------------

    df_chunk.to_pickle(output_filepath)

    print(f"\nSaved output: {output_filepath}")


if __name__ == "__main__":
    main()