#!/usr/bin/env python3

import numpy as np
import pandas as pd
import uproot
import awkward as ak
import argparse
import os


# ================================================================
# Arguments
# ================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Li9 nHits analysis for NiCf background runs per chunk"
    )

    parser.add_argument(
        "--run",
        type=int,
        required=True,
        help="NiCf run number"
    )

    parser.add_argument(
        "--chunk-id",
        type=int,
        required=True,
        help="Chunk ID"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=25000,
        help="Number of ROOT entries per chunk"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory"
    )

    parser.add_argument(
        "--base-path",
        type=str,
        required=True,
        help="Directory containing the NiCf ROOT file"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )

    return parser.parse_args()


# ================================================================
# Time RMS
# ================================================================

def default_time_rms_fun(times_in_win, t_start, window):

    if len(times_in_win) == 0:
        return 0.0, 0.0

    mean_t = np.mean(times_in_win)

    rms_t = np.sqrt(
        np.mean(
            (times_in_win - mean_t) ** 2
        )
    )

    return rms_t, mean_t


# ================================================================
# Sliding-window cluster finder
# ================================================================

def nHitstRMSTimeWindow(
    times_branch_event_arg,
    threshold_inf,
    window,
    death_window,
    time_rms_fun=default_time_rms_fun,
    rms_cut_ns=10.0,
    threshold_sup=np.inf,
):

    times = np.sort(
        np.asarray(times_branch_event_arg, float)
    )

    n = len(times)

    cand_times = []
    cand_nhits = []
    cand_trms = []

    i = 0

    while i < n:

        t_start = times[i]

        idx_end = np.searchsorted(
            times,
            t_start + window,
            side="right"
        )

        count = idx_end - i

        if count >= threshold_inf:

            times_in_win = times[i:idx_end]

            t_rms, _ = time_rms_fun(
                times_in_win,
                t_start,
                window
            )

            t_rms = float(t_rms)

            if (
                t_rms <= rms_cut_ns
                and count < threshold_sup
            ):

                cand_times.append(float(t_start))
                cand_nhits.append(count)
                cand_trms.append(t_rms)

            t_skip_until = (
                t_start
                + window
                + death_window
            )

            i = np.searchsorted(
                times,
                t_skip_until,
                side="right"
            )

        else:

            i += 1

    return (
        np.array(cand_times),
        np.array(cand_nhits),
        np.array(cand_trms)
    )


# ================================================================
# Main
# ================================================================

def main():

    args = parse_args()

    run = args.run
    chunk_id = args.chunk_id
    chunk_size = args.chunk_size
    outdir = args.outdir
    base_path = args.base_path
    verbose = args.verbose

    # ------------------------------------------------------------
    # Check allowed NiCf runs
    # ------------------------------------------------------------

    allowed_runs = [
        2437,
        2482,
        2494,
        2504,
        2507,
        2508
    ]

    if run not in allowed_runs:

        raise ValueError(
            f"Unsupported NiCf run: {run}. "
            f"Allowed runs: {allowed_runs}"
        )

    # ------------------------------------------------------------
    # Input ROOT file
    # ------------------------------------------------------------

    filename = os.path.join(
        base_path,
        f"WCTE_merged_production_R{run}_bkg.root"
    )

    output_filename = (
        f"Li9_clusters_chunk_{chunk_id}_BKG.pkl"
    )

    print(
        f"Processing NiCf BACKGROUND "
        f"for Run {run}, Chunk {chunk_id}"
    )

    if verbose:
        print(f"Opening file: {filename}")

    # ------------------------------------------------------------
    # Check input file
    # ------------------------------------------------------------

    if not os.path.isfile(filename):

        raise FileNotFoundError(
            f"Input ROOT file does not exist:\n{filename}"
        )

    # ------------------------------------------------------------
    # Open ROOT
    # ------------------------------------------------------------

    f = uproot.open(filename)

    tree = f["WCTEReadoutWindows"]

    if verbose:
        print(tree)

    # ------------------------------------------------------------
    # NiCf branches
    #
    # IMPORTANT:
    #
    # NiCf files do NOT contain:
    #
    #   hit_pmt_calibrated_times
    #
    # They contain:
    #
    #   hit_pmt_times
    #
    # ------------------------------------------------------------

    branches = [
        "window_time",
        "spill_counter",
        "hit_pmt_times",
        "hit_mpmt_slot_ids",
        "hit_pmt_position_ids",
        "hit_pmt_charges"
    ]

    if verbose:

        print("Branches requested:")

        for branch in branches:
            print(f"  {branch}")

    # ------------------------------------------------------------
    # Chunk boundaries
    # ------------------------------------------------------------

    start_entry = chunk_id * chunk_size

    stop_entry = start_entry + chunk_size

    if verbose:

        print(
            f"Reading entries "
            f"{start_entry} -> {stop_entry}"
        )

    # ------------------------------------------------------------
    # Read ROOT chunk
    # ------------------------------------------------------------

    arrays = tree.arrays(
        branches,
        entry_start=start_entry,
        entry_stop=stop_entry,
        library="ak"
    )

    if verbose:

        print(
            f"Loaded "
            f"{len(arrays.window_time)} "
            f"readout windows"
        )

    # ------------------------------------------------------------
    # Empty chunk
    # ------------------------------------------------------------

    if len(arrays.window_time) == 0:

        print(
            "No windows found in this chunk. "
            "Generating empty output structure."
        )

        df_empty = pd.DataFrame()

        os.makedirs(
            outdir,
            exist_ok=True
        )

        df_empty.to_pickle(
            os.path.join(
                outdir,
                output_filename
            )
        )

        return

    # ------------------------------------------------------------
    # Convert event-level information
    # ------------------------------------------------------------

    window_times_ns = ak.to_numpy(
        arrays.window_time
    )

    spill_ids = ak.to_numpy(
        arrays.spill_counter
    )

    # ------------------------------------------------------------
    # Flatten hit-level information
    # ------------------------------------------------------------

    hit_times_ns = ak.to_numpy(
        ak.flatten(
            arrays.hit_pmt_times
        )
    )

    slot_ids = ak.to_numpy(
        ak.flatten(
            arrays.hit_mpmt_slot_ids
        )
    )

    position_ids = ak.to_numpy(
        ak.flatten(
            arrays.hit_pmt_position_ids
        )
    )

    hit_charges = ak.to_numpy(
        ak.flatten(
            arrays.hit_pmt_charges
        )
    )

    # ------------------------------------------------------------
    # Associate each hit with its readout window
    # ------------------------------------------------------------

    hit_window_idx = ak.to_numpy(

        ak.flatten(

            ak.broadcast_arrays(

                np.arange(
                    len(window_times_ns)
                ),

                arrays.hit_pmt_times

            )[0]
        )
    )

    # ------------------------------------------------------------
    # Convert relative hit times to absolute times
    #
    # Same procedure as AmBe:
    #
    # absolute hit time =
    #     window_time + hit_time
    # ------------------------------------------------------------

    abs_hit_times_ns = (
        window_times_ns[hit_window_idx]
        + hit_times_ns
    )

    hit_spill_ids = (
        spill_ids[hit_window_idx]
    )

    if verbose:

        print(
            f"Total hits in chunk: "
            f"{len(abs_hit_times_ns)}"
        )

    # ============================================================
    # Li9 cluster selection
    # ============================================================

    window_ns = 20
    nHits_min = 15
    nHits_max = 50
    death_window = 0
    rms_cut_ns = 10.0

    # ------------------------------------------------------------
    # Containers
    # ------------------------------------------------------------

    rows = []

    spill_stats = []

    # ------------------------------------------------------------
    # Loop over spills
    # ------------------------------------------------------------

    for spill in np.unique(hit_spill_ids):

        mask_spill = (
            hit_spill_ids == spill
        )

        times_spill = (
            abs_hit_times_ns[mask_spill]
        )

        if len(times_spill) == 0:
            continue

        # --------------------------------------------------------
        # Define end of spill
        # --------------------------------------------------------

        t_end = np.max(
            times_spill
        )

        # --------------------------------------------------------
        # Search last 480 ms
        # --------------------------------------------------------

        t_start = (
            t_end
            - 0.48e9
        )

        mask_Li9 = (

            (times_spill >= t_start)
            &
            (times_spill <= t_end)

        )

        times_Li9 = (
            times_spill[mask_Li9]
        )

        if len(times_Li9) == 0:
            continue

        # --------------------------------------------------------
        # Sliding-window search
        # --------------------------------------------------------

        (
            t_window_start,
            nHits_list,
            t_rms_list
        ) = nHitstRMSTimeWindow(

            times_Li9,

            threshold_inf=nHits_min,

            threshold_sup=nHits_max,

            window=window_ns,

            death_window=death_window,

            rms_cut_ns=rms_cut_ns,

            time_rms_fun=default_time_rms_fun
        )

        num_clusters_in_spill = (
            len(t_window_start)
        )

        # --------------------------------------------------------
        # Spill statistics
        # --------------------------------------------------------

        spill_stats.append({

            "run": run,

            "spill_id": spill,

            "cluster_count":
                num_clusters_in_spill

        })

        # --------------------------------------------------------
        # Hit information
        # --------------------------------------------------------

        slot_Li9 = (
            slot_ids[mask_spill][mask_Li9]
        )

        position_Li9 = (
            position_ids[mask_spill][mask_Li9]
        )

        charge_Li9 = (
            hit_charges[mask_spill][mask_Li9]
        )

        # --------------------------------------------------------
        # Print information
        # --------------------------------------------------------

        if verbose:

            print(
                f"Spill {spill}: "
                f"{len(t_window_start)} candidates"
            )

        # --------------------------------------------------------
        # Store clusters
        # --------------------------------------------------------

        for idx in range(
            num_clusters_in_spill
        ):

            t0 = t_window_start[idx]
            nhits = nHits_list[idx]
            trms = t_rms_list[idx]
            mask_cluster = (
                (times_Li9 >= t0)
                &
                (times_Li9 < t0 + window_ns)
            )

            rows.append({

                "t_window_start_ns":
                    t0,
                "t_window_start_rel_ns":
                    t0 - t_start,
                "nHits":
                    nhits,
                "t_rms_ns":
                    trms,
                "spill_id":
                    spill,
                "nCLusters_in_spill":
                    num_clusters_in_spill,
                "hit_slot_ids":
                    slot_Li9[
                        mask_cluster
                    ].tolist(),
                "hit_position_ids":
                    position_Li9[
                        mask_cluster
                    ].tolist(),
                "hit_times_ns":
                    times_Li9[
                        mask_cluster
                    ].tolist(),
                "hit_charges":
                    charge_Li9[
                        mask_cluster
                    ].tolist()
            })

    # ============================================================
    # Create DataFrame
    # ============================================================

    df = pd.DataFrame(rows)

    if verbose:

        print(
            f"Total selected windows in chunk: "
            f"{len(df)}"
        )

    # ============================================================
    # Save output
    # ============================================================

    os.makedirs(
        outdir,
        exist_ok=True
    )

    out_file = os.path.join(
        outdir,
        output_filename
    )

    df.to_pickle(out_file)

    if verbose:

        print(
            f"Saved: {out_file}"
        )


# ================================================================
# Entry point
# ================================================================

if __name__ == "__main__":

    main()