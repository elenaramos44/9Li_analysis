import os
import glob
import argparse
import pandas as pd
import numpy as np


def main():

    parser = argparse.ArgumentParser(
        description="Merge refined chunks and apply Fiducial Volume selection."
    )

    parser.add_argument(
        "--run",
        type=int,
        required=True,
        help="Run number",
    )

    parser.add_argument(
        "--bkg",
        action="store_true",
        help="Process background sample",
    )

    # ---------------------------------------------------------
    # Fiducial Volume limits (cm)
    # ---------------------------------------------------------

    parser.add_argument(
        "--xmin",
        type=float,
        default=-20.0,
        help="Minimum x [cm]",
    )

    parser.add_argument(
        "--xmax",
        type=float,
        default=20.0,
        help="Maximum x [cm]",
    )

    parser.add_argument(
        "--ymin",
        type=float,
        default=-20.0,
        help="Minimum y [cm]",
    )

    parser.add_argument(
        "--ymax",
        type=float,
        default=20.0,
        help="Maximum y [cm]",
    )

    parser.add_argument(
        "--zmin",
        type=float,
        default=-130.0,
        help="Minimum z [cm]",
    )

    parser.add_argument(
        "--zmax",
        type=float,
        default=0.0,
        help="Maximum z [cm]",
    )

    parser.add_argument(
        "--fvtag",
        type=str,
        default="FV_1",
        help="Name of the output FV folder",
    )

    args = parser.parse_args()

    # ==========================================================
    # Directories
    # ==========================================================

    processed_dir = (
        f"/scratch/elena/9Li/results/run{args.run}/processed"
    )

    output_dir = os.path.join(
        processed_dir,
        args.fvtag,
    )

    os.makedirs(output_dir, exist_ok=True)

    # ==========================================================
    # Input files
    # ==========================================================

    if args.bkg:

        search_pattern = os.path.join(
            processed_dir,
            "Refine_Li9_clusters_chunk*_BKG.pkl",
        )

        sample_label = "BACKGROUND"

        output_filename = (
            f"Final_FV_Li9_clusters_run{args.run}_BKG.pkl"
        )

    else:

        search_pattern = os.path.join(
            processed_dir,
            "Refine_Li9_clusters_chunk*.pkl",
        )

        sample_label = "SIGNAL"

        output_filename = (
            f"Final_FV_Li9_clusters_run{args.run}.pkl"
        )

    refined_files = sorted(glob.glob(search_pattern))

    if not args.bkg:
        refined_files = [
            f for f in refined_files
            if not f.endswith("_BKG.pkl")
        ]

    if len(refined_files) == 0:

        print(
            f"No refined chunk files found for "
            f"Run {args.run} ({sample_label})"
        )

        return

    print(
        f"Found {len(refined_files)} refined chunk files "
        f"for Run {args.run} ({sample_label})"
    )

    # ==========================================================
    # Merge chunks
    # ==========================================================

    dfs = [
        pd.read_pickle(f)
        for f in refined_files
    ]

    df_all_refined = pd.concat(
        dfs,
        ignore_index=True,
    )

    print(
        f"Total refined clusters loaded: "
        f"{len(df_all_refined)}"
    )

    if df_all_refined.empty:

        print("Merged dataframe is empty.")

        return

    # ==========================================================
    # Statistics before FV cut
    # ==========================================================

    initial_clusters = len(df_all_refined)

    initial_spills = (
        set(df_all_refined["spill_id"].unique())
        if "spill_id" in df_all_refined.columns
        else set()
    )

    # ==========================================================
    # Fiducial Volume
    # ==========================================================

    x_lims = [args.xmin, args.xmax]
    y_lims = [args.ymin, args.ymax]
    z_lims = [args.zmin, args.zmax]

    print("\nApplying Fiducial Volume:")
    print(f"x = [{x_lims[0]}, {x_lims[1]}] cm")
    print(f"y = [{y_lims[0]}, {y_lims[1]}] cm")
    print(f"z = [{z_lims[0]}, {z_lims[1]}] cm")
    print(f"Output folder: {args.fvtag}")

    fv_mask = (

        (df_all_refined["v_x_fine"] >= x_lims[0])
        & (df_all_refined["v_x_fine"] <= x_lims[1])

        &

        (df_all_refined["v_y_fine"] >= y_lims[0])
        & (df_all_refined["v_y_fine"] <= y_lims[1])

        &

        (df_all_refined["v_z_fine"] >= z_lims[0])
        & (df_all_refined["v_z_fine"] <= z_lims[1])

    )

    df_final_fv = df_all_refined[fv_mask].copy()

    print(
        f"Clusters inside FV: "
        f"{len(df_final_fv)} / {len(df_all_refined)}"
    )

    # ==========================================================
    # Statistics after FV cut
    # ==========================================================

    final_clusters = len(df_final_fv)

    final_spills = (
        set(df_final_fv["spill_id"].unique())
        if "spill_id" in df_final_fv.columns
        else set()
    )

    dropped_clusters = initial_clusters - final_clusters

    lost_spills = initial_spills - final_spills

    print("\n" + "=" * 70)

    print(
        f"FV ANALYSIS : RUN {args.run} ({sample_label})"
    )

    print("=" * 70)

    print(
        f"Before FV : {initial_clusters} clusters "
        f"({len(initial_spills)} spills)"
    )

    print(
        f"After FV  : {final_clusters} clusters "
        f"({len(final_spills)} spills)"
    )

    print("-" * 70)

    print(
        f"Removed clusters : {dropped_clusters} "
        f"({100*dropped_clusters/initial_clusters:.2f}%)"
    )

    print(
        f"Removed spills   : {len(lost_spills)}"
    )

    print("=" * 70)

    # ==========================================================
    # Save
    # ==========================================================

    output_path = os.path.join(
        output_dir,
        output_filename,
    )

    df_final_fv.to_pickle(output_path)

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()