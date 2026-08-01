#!/usr/bin/env python3
import os
import argparse
import numpy as np
import uproot
import awkward as ak

def parse_args():
    parser = argparse.ArgumentParser(description="Skim WCTE ROOT files into SIGNAL and BKG using uproot with branch selection.")
    parser.add_argument("--run", type=int, required=True, help="Run number")
    parser.add_argument("--in-base", type=str, required=True, help="Base path for input data")
    parser.add_argument("--out-base", type=str, required=True, help="Base path for output processed files")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # ---------------------------------------------------------------------
    # Determinación dinámica del subdirectorio de datos según el run
    # ---------------------------------------------------------------------
    gd_runs = [2407, 2408, 2409, 2432, 2434, 2438]
    
    if args.run in gd_runs or args.run >= 2400:
        momentum_dir = "Gd/p_270"
    elif args.run in [1846, 1848]:
        momentum_dir = "p_340"
    else:
        momentum_dir = "p_260"

    input_file = os.path.join(args.in_base, momentum_dir, f"WCTE_merged_production_R{args.run}.root")
    output_dir = os.path.join(args.out_base, momentum_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    out_signal_path = os.path.join(output_dir, f"WCTE_merged_production_R{args.run}_signal.root")
    out_bkg_path = os.path.join(output_dir, f"WCTE_merged_production_R{args.run}_bkg.root")
    
    print(f"Reading: {input_file}")
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        return

    branches_to_keep = [
        "window_time",
        "spill_counter",
        "start_counter",
        "event_number",
        "readout_number",
        "hit_mpmt_slot_ids",
        "hit_pmt_position_ids",
        "hit_pmt_charges",
        "hit_pmt_calibrated_times",
        "window_data_quality_mask",
        "vme_evt_quality_bitmask",
        "vme_digi_issues_bitmask",
        "T5_HasValidHit",
        "T5_HasMultipleScintillatorsHit",
        "T5_HasOutOfTimeWindow",
        "T5_HasInTimeWindow",
        "T5_particle_nr",
        "vme_act_tagger"
    ]

    with uproot.open(input_file) as f_in:
        tree_windows = f_in["WCTEReadoutWindows"]
        tree_scalars = f_in["vme_analysis_scalar_results"]
        
        arrays_win = tree_windows.arrays(branches_to_keep, library="ak")
        arrays_sc = tree_scalars.arrays(library="ak")

    print(f"Loaded {len(arrays_win)} readout windows with essential branches.")

    # Quality Cuts and T5
    mask_quality = (
        (arrays_win["window_data_quality_mask"] == 0) &
        (arrays_win["vme_evt_quality_bitmask"] == 0) &
        (arrays_win["vme_digi_issues_bitmask"] == 0) &
        (arrays_win["T5_HasValidHit"] == True) &
        (arrays_win["T5_HasMultipleScintillatorsHit"] == False) &
        (arrays_win["T5_HasOutOfTimeWindow"] == False) &
        (arrays_win["T5_HasInTimeWindow"] == True) &
        (arrays_win["T5_particle_nr"] == 1)
    )
    
    tagger_cut = arrays_sc["act_tagger_cut"][0]
    mask_pion_event = (arrays_win["vme_act_tagger"] < tagger_cut) & mask_quality

    # Extrapolar a nivel de SPILL completo
    pion_spills = np.unique(ak.to_numpy(arrays_win["spill_counter"][mask_pion_event]))
    all_spills_vec = ak.to_numpy(arrays_win["spill_counter"])
    
    signal_window_mask = np.isin(all_spills_vec, pion_spills)
    bkg_window_mask = ~signal_window_mask

    print(f"Found {len(pion_spills)} unique spills containing pions.")
    print(f"Signal windows: {np.sum(signal_window_mask)} | Background windows: {np.sum(bkg_window_mask)}")

    # Esquemas de tipos optimizados
    tree_schema = {field: arrays_win[field].type for field in branches_to_keep}
    scalar_schema = {field: arrays_sc[field].type for field in tree_scalars.keys()}

    chunk_step = 100000

    # SIGNAL SAMPLE
    print(f"Writing Signal output in chunks: {out_signal_path}")
    signal_arrays = arrays_win[signal_window_mask]
    num_signal = len(signal_arrays)

    with uproot.recreate(out_signal_path) as f_sig:
        f_sig.mktree("WCTEReadoutWindows", tree_schema)
        f_sig.mktree("vme_analysis_scalar_results", scalar_schema)
        
        f_sig["vme_analysis_scalar_results"].extend({field: arrays_sc[field] for field in tree_scalars.keys()})
        
        for i in range(0, num_signal, chunk_step):
            chunk = signal_arrays[i : i + chunk_step]
            f_sig["WCTEReadoutWindows"].extend({field: chunk[field] for field in branches_to_keep})
            print(f"  -> Written signal entries {i} to {min(i + chunk_step, num_signal)}")

    # BACKGROUND SAMPLE
    print(f"Writing Background output in chunks: {out_bkg_path}")
    bkg_arrays = arrays_win[bkg_window_mask]
    num_bkg = len(bkg_arrays)

    with uproot.recreate(out_bkg_path) as f_bkg:
        f_bkg.mktree("WCTEReadoutWindows", tree_schema)
        f_bkg.mktree("vme_analysis_scalar_results", scalar_schema)
        
        f_bkg["vme_analysis_scalar_results"].extend({field: arrays_sc[field] for field in tree_scalars.keys()})
        
        for i in range(0, num_bkg, chunk_step):
            chunk = bkg_arrays[i : i + chunk_step]
            f_bkg["WCTEReadoutWindows"].extend({field: chunk[field] for field in branches_to_keep})
            print(f"  -> Written background entries {i} to {min(i + chunk_step, num_bkg)}")

    print("Sample separation completed successfully!")
    
if __name__ == "__main__":
    main()