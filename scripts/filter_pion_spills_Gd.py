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
    # Determinación dinámica del subdirectorio de datos según el run (Gd y estándar)
    # ---------------------------------------------------------------------
    # Bloque 1: decide momentum_dir (SIN CAMBIOS, se queda igual)
    gd_p270_runs = [2407, 2408, 2409, 2432, 2438]
    gd_p350_runs = [2374, 2379]

    if args.run in gd_p270_runs:
        momentum_dir = "Gd/p_270"
    elif args.run in gd_p350_runs:
        momentum_dir = "Gd/p_350"
    elif args.run in [1846, 1848]:
        momentum_dir = "p_340"
    else:
        momentum_dir = "p_260"

    # Bloque 2: decide input_file (SOLO TU IF/ELSE NUEVO, sin el bloque viejo detrás)
    if args.run == 2379:
        input_file = os.path.join("/scratch/elena", momentum_dir, f"WCTE_merged_production_R{args.run}.root")
    else:
        input_file = os.path.join(args.in_base, momentum_dir, f"WCTE_merged_production_R{args.run}.root")

    output_dir = os.path.join(args.out_base, momentum_dir)
    os.makedirs(output_dir, exist_ok=True)

    out_signal_path = os.path.join(output_dir, f"WCTE_merged_production_R{args.run}_signal.root")
    out_bkg_path = os.path.join(output_dir, f"WCTE_merged_production_R{args.run}_bkg.root")

    print(f"Reading: {input_file}")
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        return

    desired_branches = [
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
        "T5_n_main_bunch_particles",
        "vme_act_tagger",
        "vme_act_eveto",
        "vme_t0_time",
        "vme_t1_time",
        "vme_t4_time"
    ]

    with uproot.open(input_file) as f_in:
        tree_windows = f_in["WCTEReadoutWindows"]
        tree_scalars = f_in["vme_analysis_scalar_results"]

        available_branches = tree_windows.keys()
        branches_to_keep = [b for b in desired_branches if b in available_branches]

        arrays_win = tree_windows.arrays(branches_to_keep, library="ak")
        arrays_sc = tree_scalars.arrays(library="ak")

    print(f"Loaded {len(arrays_win)} readout windows with essential branches.")

    # ------------------------------------------------------------------
    # AVISO EXPLÍCITO si faltan ramas críticas, en vez de fallback mudo
    # ------------------------------------------------------------------
    if "T5_HasValidHit" not in arrays_win.fields:
        print(f"*** WARNING: Run {args.run} lacks full T5 branches. "
              f"good_mask will be WEAKER than standard runs (only particle "
              f"multiplicity cut applied). Verify this is expected. ***")

    if "act_tagger_cut" not in arrays_sc.fields:
        print(f"*** WARNING: Run {args.run} lacks 'act_tagger_cut' in "
              f"vme_analysis_scalar_results. Falling back to tagger_cut=0.0, "
              f"which may reject nearly all events as non-pion. Verify this "
              f"is expected. ***")

    # 1. Event Quality Checks
    data_quality = (arrays_win["window_data_quality_mask"] == 0) if "window_data_quality_mask" in arrays_win.fields else True
    evt_quality = (arrays_win["vme_evt_quality_bitmask"] == 0) if "vme_evt_quality_bitmask" in arrays_win.fields else True
    digi_issues = (arrays_win["vme_digi_issues_bitmask"] == 0) if "vme_digi_issues_bitmask" in arrays_win.fields else True

    # 2. T5 Selection
    if "T5_HasValidHit" in arrays_win.fields:
        valid_hit = (arrays_win["T5_HasValidHit"] == True)
        mult_scint = (arrays_win["T5_HasMultipleScintillatorsHit"] == False)
        out_of_time = (arrays_win["T5_HasOutOfTimeWindow"] == False)
        in_time = (arrays_win["T5_HasInTimeWindow"] == True)
        particle_nr = (arrays_win["T5_particle_nr"] == 1)
    else:
        valid_hit = True
        mult_scint = True
        out_of_time = True
        in_time = True
        particle_nr = (arrays_win["T5_n_main_bunch_particles"] == 1) if "T5_n_main_bunch_particles" in arrays_win.fields else False

    # 3. T0 / T1 / T4 Coincidence Checks
    T0_hit = ~np.isnan(ak.to_numpy(arrays_win["vme_t0_time"])) if "vme_t0_time" in arrays_win.fields else True
    T1_hit = ~np.isnan(ak.to_numpy(arrays_win["vme_t1_time"])) if "vme_t1_time" in arrays_win.fields else True
    T4_hit = ~np.isnan(ak.to_numpy(arrays_win["vme_t4_time"])) if "vme_t4_time" in arrays_win.fields else True

    good_mask = (
        data_quality
        & evt_quality
        & digi_issues
        & valid_hit
        & mult_scint
        & out_of_time
        & in_time
        & particle_nr
        & T0_hit
        & T1_hit
        & T4_hit
    )

    # 4. Two-step PID Selection
    eveto_cut = float(ak.to_numpy(arrays_sc["act_eveto_cut"])[0]) if "act_eveto_cut" in arrays_sc.fields else 3.92
    mask_no_electrons = (arrays_win["vme_act_eveto"] < eveto_cut) if "vme_act_eveto" in arrays_win.fields else True

    # NOTA: default cambiado de 0.0 a np.inf. Un default de 0.0 rechazaba
    # casi todos los eventos como "no-pion" si faltaba act_tagger_cut,
    # colapsando la señal silenciosamente.
    tagger_cut = float(ak.to_numpy(arrays_sc["act_tagger_cut"])[0]) if "act_tagger_cut" in arrays_sc.fields else np.inf
    mask_pion_event = good_mask & mask_no_electrons & (arrays_win["vme_act_tagger"] < tagger_cut if "vme_act_tagger" in arrays_win.fields else True)

    # 5. Spill-level Classification (Excluding empty/dead spills from background)
    all_spills_vec = ak.to_numpy(arrays_win["spill_counter"])

    valid_beam_spills = np.unique(ak.to_numpy(arrays_win["spill_counter"][good_mask]))
    pion_spills = np.unique(ak.to_numpy(arrays_win["spill_counter"][mask_pion_event]))

    signal_window_mask = np.isin(all_spills_vec, pion_spills)

    non_pion_beam_spills = np.setdiff1d(valid_beam_spills, pion_spills)
    bkg_window_mask = np.isin(all_spills_vec, non_pion_beam_spills)

    print(f"Found {len(pion_spills)} unique spills containing true pions (without e- contamination).")
    print(f"Found {len(non_pion_beam_spills)} unique valid non-pion beam spills for background.")
    print(f"Signal windows: {np.sum(signal_window_mask)} | Background windows: {np.sum(bkg_window_mask)}")

    # Chunk-based writing scheme
    tree_schema = {field: arrays_win[field].type for field in branches_to_keep}
    scalar_schema = {field: arrays_sc[field].type for field in tree_scalars.keys()}
    chunk_step = 20000

    # Write SIGNAL_SAMPLE
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

    # Write BACKGROUND_SAMPLE
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