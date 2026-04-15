#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse
import os
import sys

sys.path.append("/scratch/elena/9Li/scripts")

import functions_bonsai


# =========================================================
# ARGUMENTS
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="BONSAI vertex reconstruction for Li9 clusters"
    )

    parser.add_argument("--csv", type=str, required=True, help="Input CSV with clusters")
    parser.add_argument("--outdir", type=str, required=True, help="Output folder")
    parser.add_argument("--verbose", action="store_true", help="Print progress")

    return parser.parse_args()


# =========================================================
# MAIN RECONSTRUCTION FUNCTION
# =========================================================

def run_bonsai_on_cluster(row, bonsai, lookup):
    """
    Run BONSAI on one cluster (row of dataframe)
    """

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    times = np.asarray(row["hit_times_ns"], dtype=float)
    charges = np.asarray(row["hit_charges"], dtype=float)
    mpmt_ids = np.asarray(row["hit_card_ids"], dtype=int)
    pmt_ids = np.asarray(row["hit_position_ids"], dtype=int)

    # -----------------------------
    # CLEANING (PHYSICS LAYER)
    # -----------------------------

    # charge cut (noise removal)
    mask = charges > 0.5

    times = times[mask]
    charges = charges[mask]
    mpmt_ids = mpmt_ids[mask]
    pmt_ids = pmt_ids[mask]

    # safety check
    if len(times) < 5:
        return np.nan, np.nan, np.nan

    # time shift (ONLY HERE, NOT IN BONSAI MODULE)
    times = times - np.min(times)

    # -----------------------------
    # RUN BONSAI
    # -----------------------------
    vertex = functions_bonsai.run_BONSAI_candidate(
        bonsai=bonsai,
        lookup=lookup,
        times=times,
        charges=charges,
        mpmt=mpmt_ids,
        pmt=pmt_ids,
        time_recenter=False  # already done above
    )

    # -----------------------------
    # SAFE OUTPUT HANDLING
    # -----------------------------
    if not vertex.get("success", False):
        return np.nan, np.nan, np.nan

    return vertex["x"], vertex["y"], vertex["z"]


# =========================================================
# MAIN
# =========================================================

def main():
    args = parse_args()

    # -----------------------------
    # IMPORT BONSAI ENVIRONMENT
    # -----------------------------
    from functions_bonsai import (
        init_bonsai_environment,
        get_geo_mapping,
        build_lookup_table
    )

    bonsai = init_bonsai_environment()
    geo = get_geo_mapping()
    lookup = build_lookup_table(geo)

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    df = pd.read_csv(
        args.csv,
        converters={
            "hit_times_ns": eval,
            "hit_charges": eval,
            "hit_card_ids": eval,
            "hit_slot_ids": eval,
            "hit_channel_ids": eval,
            "hit_position_ids": eval,
        },
    )

    if args.verbose:
        print(f"Loaded {len(df)} clusters from {args.csv}")

    # -----------------------------
    # OUTPUT ARRAYS
    # -----------------------------
    vertex_x = np.full(len(df), np.nan)
    vertex_y = np.full(len(df), np.nan)
    vertex_z = np.full(len(df), np.nan)

    # -----------------------------
    # LOOP
    # -----------------------------
    for i, row in df.iterrows():

        if args.verbose and i % 100 == 0:
            print(f"Processing cluster {i}/{len(df)}")

        try:
            x, y, z = run_bonsai_on_cluster(row, bonsai, lookup)

            vertex_x[i] = x
            vertex_y[i] = y
            vertex_z[i] = z

        except Exception as e:
            if args.verbose:
                print(f"BONSAI failed at cluster {i}: {e}")

    # -----------------------------
    # SAVE OUTPUT
    # -----------------------------
    df["vertex_x"] = vertex_x
    df["vertex_y"] = vertex_y
    df["vertex_z"] = vertex_z

    os.makedirs(args.outdir, exist_ok=True)

    basename = os.path.basename(args.csv).replace(".csv", "_bonsai.csv")
    out_file = os.path.join(args.outdir, basename)

    df.to_csv(out_file, index=False)

    if args.verbose:
        print(f"Saved BONSAI output to {out_file}")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()