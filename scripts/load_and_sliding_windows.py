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
        t_hit = times_branch_event[i]                                                       #cada hit define el inicio de una ventna
        idx_end = np.searchsorted(times_branch_event, t_hit + window, side='right')         #busca el índice del primer hit fuera de la ventana
        if len(charge_branch_event) != 0:
            count = max(charge_branch_event[i:idx_end])  #podríamos usar charge
        else:
            count = idx_end - i                          #usamos counts = number of hits

        if count > threshold_inf and count < threshold_sup:       #cluster selection cuts
            threshold_times.append(t_hit)
            nhits_range.append(count)                             #guardamos cluster si se cumple la condición
            
            i = np.searchsorted(times_branch_event, t_hit + window + death_window, side='right')         #if candidtae, jump to the next hit after the sw   
        else:
            i += 1                                                                                       #if not, just go to the next hit and open new sw

    return np.array(threshold_times), np.array(nhits_range)


def main():
    args = parse_args()
    run = args.run
    chunk_id = args.chunk_id
    chunk_size = args.chunk_size
    outdir = args.outdir
    base_path = args.base_path
    verbose = args.verbose

    filename = os.path.join(base_path, f"WCTE_offline_R{run}S0_VME_matched.root")
    if verbose:
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
    

    # parallelization --> chunks
    start_entry = chunk_id * chunk_size
    stop_entry = start_entry + chunk_size

    arrays = tree.arrays(branches, entry_start=start_entry, entry_stop=stop_entry, library="ak")

    if verbose:
        print(f"Loaded {len(arrays.window_time)} readout windows")

    window_times_ns = ak.to_numpy(arrays.window_time)
    spill_ids = ak.to_numpy(arrays.spill_counter)

    # flatten hits for relative times within each window
    hit_times_ns = ak.to_numpy(ak.flatten(arrays.hit_pmt_calibrated_times))
    hit_card_ids = ak.to_numpy(ak.flatten(arrays.hit_mpmt_card_ids))
    hit_slot_ids = ak.to_numpy(ak.flatten(arrays.hit_mpmt_slot_ids))
    hit_channel_ids = ak.to_numpy(ak.flatten(arrays.hit_pmt_channel_ids))
    hit_position_ids = ak.to_numpy(ak.flatten(arrays.hit_pmt_position_ids))
    hit_charges = ak.to_numpy(ak.flatten(arrays.hit_pmt_charges))


    hit_window_idx = ak.to_numpy(                                               #asigna sw correspondientes a los hits
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

    
    #sliding window parameters
    window_ns = 20
    nHits_min = 15
    nHits_max = 40
    death_window = 0  
    boundary_cut = nHits_max - 1


    rows = []
    spill_stats = [] # To store cluster counts per spill


    # Loop per spill
    for spill in np.unique(hit_spill_ids):

        mask_spill = hit_spill_ids == spill
        times_spill = abs_hit_times_ns[mask_spill]

        if len(times_spill) == 0:
            continue

        # define Li9 window
        t_end = np.max(times_spill)
        t_start = t_end - 0.48e9  # 0.5 s in ns

        mask_Li9 = (times_spill >= t_start) & (times_spill <= t_end)
        times_Li9 = times_spill[mask_Li9]


        if len(times_Li9) == 0:
            continue

        # sliding window on THIS SPILL only
        t_window_start, nHits_list = nHitsTimeWindow(
            times_Li9,
            threshold_inf=nHits_min,
            threshold_sup=nHits_max,
            window=window_ns,
            death_window=death_window,
            charge_branch_event=[]   # explícito para evitar ambigüedad futura
        )

        valid_indices = [i for i, nh in enumerate(nHits_list) if nh < boundary_cut]
        num_clusters_in_spill = len(valid_indices)    #activity metric
        
        # Save spill activity info
        spill_stats.append({
            "run": run,
            "spill_id": spill,
            "cluster_count": num_clusters_in_spill
        })

        card_Li9 = hit_card_ids[mask_spill][mask_Li9]
        slot_Li9 = hit_slot_ids[mask_spill][mask_Li9]
        channel_Li9 = hit_channel_ids[mask_spill][mask_Li9]
        position_Li9 = hit_position_ids[mask_spill][mask_Li9]
        charge_Li9 = hit_charges[mask_spill][mask_Li9]

        if verbose:
            print(f"Spill {spill}: {len(t_window_start)} candidates")


        #save results
        for idx in valid_indices:
            t0 = t_window_start[idx]
            nhits = nHits_list[idx]
            mask_cluster = (times_Li9 >= t0) & (times_Li9 < t0 + window_ns)
            
            rows.append({
                "t_window_start_ns": t0,                   #abs time of the run
                "t_window_start_rel_ns": t0 - t_start,     #rel time within the sw
                "nHits": nhits,
                "spill_id": spill,
                "nCLusters_in_spill": num_clusters_in_spill,

                "hit_card_ids": card_Li9[mask_cluster].tolist(),
                "hit_slot_ids": slot_Li9[mask_cluster].tolist(),
                "hit_channel_ids": channel_Li9[mask_cluster].tolist(),
                "hit_position_ids": position_Li9[mask_cluster].tolist(),
                "hit_times_ns": times_Li9[mask_cluster].tolist(),
                "hit_charges": charge_Li9[mask_cluster].tolist()
            })

 
    df = pd.DataFrame(rows)

    if verbose:
        print(f"Total selected windows in chunk: {len(df)}")


    os.makedirs(outdir, exist_ok=True)
    out_file = os.path.join(outdir, f"Li9_clusters_chunk_{chunk_id}.csv")
    df.to_csv(out_file, index=False)

    if verbose:
        print(f"Saved: {out_file}")

if __name__ == "__main__":
    main()