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
    parser.add_argument("--entry-start", type=int, required=True)
    parser.add_argument("--entry-stop", type=int, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--base-path", type=str, required=True)
    # --- MODIFICACIÓN: Añadida bandera booleana para discriminar la muestra ---
    parser.add_argument("--bkg", action="store_true", help="Process background root file instead of signal")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

def default_time_rms_fun(times_in_win, t_start, window):
    if len(times_in_win) == 0:
        return 0.0, 0.0
    mean_t = np.mean(times_in_win)
    rms_t = np.sqrt(np.mean((times_in_win - mean_t) ** 2))
    return rms_t, mean_t

def nHitstRMSTimeWindow(
    times_branch_event_arg,
    threshold_inf,
    window,
    death_window,
    time_rms_fun=default_time_rms_fun,
    rms_cut_ns=10.0,
    threshold_sup=np.inf,
):
    times = np.sort(np.asarray(times_branch_event_arg, float))
    n = len(times)

    cand_times = []
    cand_nhits = []
    cand_trms  = []

    i = 0
    while i < n:
        t_start = times[i]
        idx_end = np.searchsorted(times, t_start + window, side='right')
        count = idx_end - i

        if count >= threshold_inf:
            times_in_win = times[i:idx_end]
            t_rms, _ = time_rms_fun(times_in_win, t_start, window)
            t_rms = float(t_rms)

            if (t_rms <= rms_cut_ns) and (count < threshold_sup):
                cand_times.append(float(t_start))
                cand_nhits.append(count)
                cand_trms.append(t_rms)

            t_skip_until = t_start + window + death_window
            i = np.searchsorted(times, t_skip_until, side='right')
        else:
            i += 1

    return np.array(cand_times), np.array(cand_nhits), np.array(cand_trms)


def main():
    args = parse_args()
    run = args.run
    chunk_id = args.chunk_id
    chunk_size = args.chunk_size
    outdir = args.outdir
    base_path = args.base_path
    verbose = args.verbose

    # --- MODIFICACIÓN: Selección dinámica del archivo ROOT de entrada y de la etiqueta de salida ---
    if args.bkg:
        filename = os.path.join(base_path, f"WCTE_merged_production_R{run}_bkg.root")
        output_filename = f"Li9_clusters_chunk_{chunk_id}_BKG.pkl"
        sample_label = "BACKGROUND"
    else:
        filename = os.path.join(base_path, f"WCTE_merged_production_R{run}_signal.root")
        output_filename = f"Li9_clusters_chunk_{chunk_id}.pkl"
        sample_label = "SIGNAL"

    print(f"Processing {sample_label} for Run {run}, Chunk {chunk_id}")
    if verbose:
        print(f"Opening file: {filename}")

    f = uproot.open(filename)
    tree = f["WCTEReadoutWindows"]

    branches = [
        "window_time",
        "spill_counter",
        "hit_pmt_calibrated_times",
        "hit_mpmt_slot_ids",
        "hit_pmt_position_ids",
        "hit_pmt_charges"
    ]
    
    #start_entry = chunk_id * chunk_size    #old fixed version!
    #stop_entry = start_entry + chunk_size

    start_entry = args.entry_start
    stop_entry = args.entry_stop

    arrays = tree.arrays(branches, entry_start=start_entry, entry_stop=stop_entry, library="ak")


    if verbose:
        print(f"Loaded {len(arrays.window_time)} readout windows")

    if len(arrays.window_time) == 0:
        print("No windows found in this chunk. Generating empty output structure.")
        df_empty = pd.DataFrame()
        os.makedirs(outdir, exist_ok=True)
        df_empty.to_pickle(os.path.join(outdir, output_filename))
        return

    window_times_ns = ak.to_numpy(arrays.window_time)
    spill_ids = ak.to_numpy(arrays.spill_counter)

    hit_times_ns = ak.to_numpy(ak.flatten(arrays.hit_pmt_calibrated_times))
    slot_ids = ak.to_numpy(ak.flatten(arrays.hit_mpmt_slot_ids))
    position_ids = ak.to_numpy(ak.flatten(arrays.hit_pmt_position_ids))
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

    if verbose:
        print(f"Total hits in chunk: {len(abs_hit_times_ns)}")
    
    window_ns = 20
    nHits_min = 15
    nHits_max = 50
    death_window = 0  
    rms_cut_ns = 10.0

    rows = []
    spill_stats = []

    for spill in np.unique(hit_spill_ids):
        mask_spill = hit_spill_ids == spill
        times_spill = abs_hit_times_ns[mask_spill]

        if len(times_spill) == 0:
            continue

        t_end = np.max(times_spill)
        t_start = t_end - 0.48e9  

        mask_Li9 = (times_spill >= t_start) & (times_spill <= t_end)
        times_Li9 = times_spill[mask_Li9]

        if len(times_Li9) == 0:
            continue

        t_window_start, nHits_list, t_rms_list = nHitstRMSTimeWindow(
            times_Li9,
            threshold_inf=nHits_min,
            threshold_sup=nHits_max,
            window=window_ns,
            death_window=death_window,
            rms_cut_ns=rms_cut_ns,
            time_rms_fun=default_time_rms_fun
        )

        num_clusters_in_spill = len(t_window_start)    
        
        spill_stats.append({
            "run": run,
            "spill_id": spill,
            "cluster_count": num_clusters_in_spill
        })

        slot_Li9 = slot_ids[mask_spill][mask_Li9]
        position_Li9 = position_ids[mask_spill][mask_Li9]
        charge_Li9 = hit_charges[mask_spill][mask_Li9]

        if verbose:
            print(f"Spill {spill}: {len(t_window_start)} candidates")

        for idx in range(num_clusters_in_spill):
            t0 = t_window_start[idx]
            nhits = nHits_list[idx]
            trms = t_rms_list[idx]
            mask_cluster = (times_Li9 >= t0) & (times_Li9 < t0 + window_ns)
            
            rows.append({
                "t_window_start_ns": t0,                   
                "t_window_start_rel_ns": t0 - t_start,     
                "nHits": nhits,
                "t_rms_ns": trms,                          
                "spill_id": spill,
                "nCLusters_in_spill": num_clusters_in_spill,

                "hit_slot_ids": slot_Li9[mask_cluster].tolist(),
                "hit_position_ids": position_Li9[mask_cluster].tolist(),
                "hit_times_ns": times_Li9[mask_cluster].tolist(),
                "hit_charges": charge_Li9[mask_cluster].tolist()
            })

    df = pd.DataFrame(rows)

    if verbose:
        print(f"Total selected windows in chunk: {len(df)}")

    # --- MODIFICACIÓN: Se guarda como archivo .pkl usando to_pickle ---
    os.makedirs(outdir, exist_ok=True)
    out_file = os.path.join(outdir, output_filename)
    df.to_pickle(out_file)

    if verbose:
        print(f"Saved: {out_file}")

if __name__ == "__main__":
    main()