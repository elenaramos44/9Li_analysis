#!/usr/bin/env python3
"""
Filter multiple WCTE ROOT files post-spill, apply Nhit and detector cuts,
and save results as NPZ files (one per ROOT file).
"""

import uproot
import awkward as ak
import numpy as np
import argparse
import os

# ---------------- Command-line arguments ----------------
parser = argparse.ArgumentParser(description="Filter post-spill WCTE hits for multiple ROOT files")
parser.add_argument("--root-files", type=str, nargs="+", required=True,
                    help="List of ROOT files to process in this job")
parser.add_argument("--outdir", type=str, required=True,
                    help="Directory to save NPZ files")
parser.add_argument("--spill_threshold", type=int, default=300,
                    help="Hits per window threshold to detect spills")
parser.add_argument("--post_spill_window", type=float, default=0.5,
                    help="Time window after spill end [s] to select hits")
parser.add_argument("--nhit_min", type=int, default=15,
                    help="Minimum number of hits per post-spill window")
parser.add_argument("--nhit_max", type=int, default=40,
                    help="Maximum number of hits per post-spill window")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# ---------------- Branches to load ----------------
branches = [
    "window_time",
    "start_counter",
    "spill_counter",
    "readout_number",
    "hit_mpmt_card_ids",
    "hit_pmt_channel_ids",
    "hit_mpmt_slot_ids",
    "hit_pmt_position_ids",
    "hit_pmt_charges",
    "hit_pmt_times",
    "beamline_pmt_qdc_charges",
    "beamline_pmt_tdc_times",
    "beamline_pmt_qdc_ids",
    "beamline_pmt_tdc_ids"
]

# ---------------- Process each ROOT file ----------------
for ROOT_FILE in args.root_files:
    print(f"\nProcessing {ROOT_FILE} ...")
    with uproot.open(ROOT_FILE) as file:
        arrays = file["WCTEReadoutWindows"].arrays(branches, library="ak")

    # Sort by window_time
    arrays = arrays[ak.argsort(arrays["window_time"])]

    # Detect beam spills
    hits_per_window = ak.num(arrays["hit_pmt_charges"])
    times = arrays["window_time"] / 1e9  # ns → s

    beam_times = np.sort(times[hits_per_window > args.spill_threshold])
    spill_edges = []

    if len(beam_times) > 0:
        min_gap = 0.5
        current_start = beam_times[0]
        current_end = beam_times[0]

        for t in beam_times[1:]:
            if t - current_end <= min_gap:
                current_end = t
            else:
                spill_edges.append((current_start, current_end))
                current_start = t
                current_end = t
        spill_edges.append((current_start, current_end))

    # ---------------- Select post-spill hits ----------------
    post_spill_mask = ak.zeros_like(times, dtype=bool)
    for start, end in spill_edges:
        mask = (times >= end) & (times <= end + args.post_spill_window)
        post_spill_mask = post_spill_mask | mask

    fields = arrays.fields
    filtered_hits = {field: arrays[field][post_spill_mask] for field in fields}
    print("Number of post-spill readout windows:", len(filtered_hits["window_time"]))

    # ---------------- Apply Nhit cut ----------------
    nhits_post_spill = ak.num(filtered_hits["hit_pmt_charges"])
    nhit_mask = (nhits_post_spill >= args.nhit_min) & (nhits_post_spill <= args.nhit_max)
    filtered_hits = {field: filtered_hits[field][nhit_mask] for field in fields}
    print("Windows passing Nhit cut:", len(filtered_hits["window_time"]))

    # ---------------- Apply detector cuts ----------------
    slot_mask = ak.all(filtered_hits["hit_mpmt_slot_ids"] != -1, axis=1)
    card_mask = ak.all(filtered_hits["hit_mpmt_card_ids"] < 130, axis=1)
    detector_mask = slot_mask & card_mask
    filtered_hits = {field: filtered_hits[field][detector_mask] for field in fields}
    print("Windows passing Nhit + detector cuts:", len(filtered_hits["window_time"]))

    # ---------------- Save NPZ ----------------
    out_file = os.path.join(args.outdir, os.path.basename(ROOT_FILE).replace(".root", ".npz"))
    np.savez_compressed(out_file, **{k: ak.to_numpy(v) for k, v in filtered_hits.items()})
    print(f"Saved {out_file}")