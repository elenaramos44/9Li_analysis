#!/usr/bin/env python3
import numpy as np
import pandas as pd
import uproot
import awkward as ak
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Li9 nHits analysis per chunk")
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=25000)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--base-path", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

def nHitsTimeWindow(times_branch_event_arg, threshold_inf, window, death_window=0, charge_branch_event=[], threshold_sup=np.inf):
    """
    Sliding window optimized: si cumple el criterio de hits, se guarda y la siguiente ventana
    empieza justo después de terminar la actual (no solapamientos). Si no cumple, avanzamos al siguiente hit.
    uso de np.searchsorted para saltar directamente al final de la ventana.
    """
    times_branch_event = np.sort(times_branch_event_arg.copy())
    threshold_times = []
    nhits_range = []
    n = len(times_branch_event)
    i = 0

    while i < n:
        t_hit = times_branch_event[i]
        idx_end = np.searchsorted(times_branch_event, t_hit + window, side='right')

        if len(charge_branch_event) != 0:
            count = max(charge_branch_event[i:idx_end])
        else:
            count = idx_end - i

        if count > threshold_inf and count < threshold_sup:
            threshold_times.append(t_hit)
            nhits_range.append(count)

            i = np.searchsorted(times_branch_event, t_hit + window + death_window, side='right')
        else:
            i += 1

    return np.array(threshold_times), np.array(nhits_range)


def main():

    args = parse_args()

    filename = os.path.join(args.base_path, f"WCTE_offline_R{args.run}S0_VME_matched.root")

    if args.verbose:
        print(f"Opening file: {filename}")

    f = uproot.open(filename)
    tree = f["WCTEReadoutWindows"]

    branches = [
        "window_time",
        "spill_counter",
        "hit_pmt_calibrated_times",
        "hit_mpmt_card_ids",
        "hit_pmt_channel_ids",
        "hit_mpmt_slot_ids",
        "hit_pmt_position_ids",
        "hit_pmt_charges"
    ]

    arrays = tree.arrays(
        branches,
        entry_start=args.chunk_id * args.chunk_size,
        entry_stop=(args.chunk_id + 1) * args.chunk_size,
        library="ak"
    )

    window_times_ns = ak.to_numpy(arrays.window_time)
    spill_ids = ak.to_numpy(arrays.spill_counter)

    hit_times_ns = ak.to_numpy(ak.flatten(arrays.hit_pmt_calibrated_times))
    hit_card_ids = ak.to_numpy(ak.flatten(arrays.hit_mpmt_card_ids))
    hit_slot_ids = ak.to_numpy(ak.flatten(arrays.hit_mpmt_slot_ids))
    hit_channel_ids = ak.to_numpy(ak.flatten(arrays.hit_pmt_channel_ids))
    hit_position_ids = ak.to_numpy(ak.flatten(arrays.hit_pmt_position_ids))
    hit_charges = ak.to_numpy(ak.flatten(arrays.hit_pmt_charges))

    hit_window_idx = ak.to_numpy(
        ak.flatten(
            ak.broadcast_arrays(
                np.arange(len(window_times_ns)),
                arrays.hit_pmt_calibrated_times
            )[0]
        )
    )

    abs_hit_times_ns = window_times_ns[hit_window_idx] + hit_times_ns
    hit_spill_ids = spill_ids[hit_window_idx]

    window_ns = 25
    nHits_min = 15
    nHits_max = 40
    death_window = 0

    rows = []

    for spill in np.unique(hit_spill_ids):

        mask_spill = (hit_spill_ids == spill)
        times_spill = abs_hit_times_ns[mask_spill]

        if len(times_spill) == 0:
            continue

        t_end = np.max(times_spill)
        t_start = t_end - 0.5e9

        mask_Li9 = (times_spill >= t_start) & (times_spill <= t_end)

        # -------------------------
        # BONSAI-SAFE HIT TABLE
        # -------------------------
        hits = pd.DataFrame({
            "t": times_spill[mask_Li9],
            "q": hit_charges[mask_spill][mask_Li9],
            "card": hit_card_ids[mask_spill][mask_Li9],
            "slot": hit_slot_ids[mask_spill][mask_Li9],
            "channel": hit_channel_ids[mask_spill][mask_Li9],
            "pos": hit_position_ids[mask_spill][mask_Li9],
        }).sort_values("t").reset_index(drop=True)

        if len(hits) == 0:
            continue

        t_window_start, nHits_list = nHitsTimeWindow(
            hits["t"].to_numpy(),
            threshold_inf=nHits_min,
            threshold_sup=nHits_max,
            window=window_ns,
            death_window=death_window,
            charge_branch_event=hits["q"].to_numpy()
        )

        if args.verbose:
            print(f"Spill {spill}: {len(t_window_start)} candidates")

        for t0, nhits in zip(t_window_start, nHits_list):

            mask_cluster = (hits["t"].to_numpy() >= t0) & (hits["t"].to_numpy() < t0 + window_ns)
            h = hits[mask_cluster]

            rows.append({
                "t_window_start_ns": t0,
                "t_window_start_rel_ns": t0 - t_start,
                "nHits": nhits,
                "spill_id": spill,

                "hit_card_ids": h["card"].tolist(),
                "hit_slot_ids": h["slot"].tolist(),
                "hit_channel_ids": h["channel"].tolist(),
                "hit_position_ids": h["pos"].tolist(),
                "hit_times_ns": h["t"].tolist(),
                "hit_charges": h["q"].tolist()
            })

    df = pd.DataFrame(rows)

    if args.verbose:
        print(f"Total selected windows: {len(df)}")

    os.makedirs(args.outdir, exist_ok=True)

    out_file = os.path.join(
        args.outdir,
        f"Li9_clusters_chunk_{args.chunk_id}.csv"
    )

    df.to_csv(out_file, index=False)

    if args.verbose:
        print(f"Saved: {out_file}")

if __name__ == "__main__":
    main()