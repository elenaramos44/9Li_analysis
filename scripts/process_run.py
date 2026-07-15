#!/usr/bin/env python3
import argparse
import os
import re
import sys
import awkward as ak
import numpy as np
import pandas as pd
import uproot

# ==============================================================================
# COMMAND LINE ARGUMENTS CONTROLLER
# ==============================================================================
parser = argparse.ArgumentParser(
    description="Processes a WCTE ROOT file to calculate isotope production at Run level."
)
parser.add_argument(
    "--input", required=True, type=str, help="Path to the .root file to process"
)
parser.add_argument(
    "--outdir", required=True, type=str, help="Directory where the .csv will be saved"
)
args = parser.parse_args()

filename = args.input
outdir = args.outdir

os.makedirs(outdir, exist_ok=True)

# ==============================================================================
# CONFIGURATION AND PHYSICAL PARAMETERS (DYNAMIC DETECTION)
# ==============================================================================
# Detect beam momentum based on the parent folder (p_340 -> -340, p_260 -> -260)
beam_momentum = -340  # Default value just in case
if "p_260" in filename:
    beam_momentum = -260
elif "p_340" in filename:
    beam_momentum = -340

TAU_12B = 30.0
TAU_9LI = 257.0
TAU_16N = 7130.0

t_start_win = 20.0
t_threshold = 50.0
t_end_win = 500.0

# Dynamically extract Run number from the filename
match = re.search(r"R(\d{4})\.root", os.path.basename(filename))
run_number = match.group(1) if match else "Unknown"

print(f">>> Launching Run {run_number} | Beam p: {beam_momentum} MeV/c...")

# ==============================================================================
# LOAD BRANCHES 
# ==============================================================================
window_branches = [
    "window_data_quality_mask",
    "vme_evt_quality_bitmask",
    "vme_digi_issues_bitmask",
    "T5_HasValidHit",
    "T5_HasMultipleScintillatorsHit",
    "T5_HasOutOfTimeWindow",
    "T5_HasInTimeWindow",
    "T5_particle_nr",
    "vme_act_tagger",
    "window_time",
    "spill_counter",
    "event_number",
    "readout_number",
]

scalar_branches = ["act_tagger_cut"]

with uproot.open(filename) as file:
    tree_windows = file["WCTEReadoutWindows"]
    tree_scalars = file["vme_analysis_scalar_results"]

    arrays_windows = tree_windows.arrays(window_branches, library="ak")
    arrays_scalars = tree_scalars.arrays(scalar_branches, library="ak")

# ==============================================================================
# FILTERING AND QUALITY CHECKS
# ==============================================================================
# 1. Apply general detector quality checks to the data windows
good_mask = (
    (arrays_windows["window_data_quality_mask"] == 0)
    & (arrays_windows["vme_evt_quality_bitmask"] == 0)
    & (arrays_windows["vme_digi_issues_bitmask"] == 0)
    & (arrays_windows["T5_HasValidHit"] == True)
    & (arrays_windows["T5_HasMultipleScintillatorsHit"] == False)
    & (arrays_windows["T5_HasOutOfTimeWindow"] == False)
    & (arrays_windows["T5_HasInTimeWindow"] == True)
    & (arrays_windows["T5_particle_nr"] == 1)
)

filtered_windows = arrays_windows[good_mask]

# CORRECTION: Count total spills that successfully passed the detector quality checks
n_spills_total = float(len(np.unique(filtered_windows.spill_counter))) if len(filtered_windows) > 0 else 1.0

# 2. Apply the act_tagger threshold condition to select verified pions
eveto_cut_value = ak.max(arrays_scalars["act_tagger_cut"])
pion_mask = filtered_windows["vme_act_tagger"] < eveto_cut_value
pion_events = filtered_windows[pion_mask]

if len(pion_events) == 0:
    print(
        f" [NOTICE] Zero pions detected in Run {run_number}. Exiting cleanly."
    )
    sys.exit(0)

target_branches = [
    "window_time",
    "spill_counter",
    "event_number",
    "readout_number",
]
df_pion_events = ak.to_dataframe(pion_events[target_branches]).reset_index(
    drop=True
)

# ==============================================================================
# SPILL TEMPORAL ALIGNMENT
# ==============================================================================
spill_timing_map = {}
unique_spills = np.unique(arrays_windows.spill_counter)

for spill_id in unique_spills:
    spill_raw_mask = arrays_windows.spill_counter == spill_id
    window_times_us = ak.to_numpy(arrays_windows[spill_raw_mask].window_time) / 1000

    if len(window_times_us) > 0:
        t_start_spill_us = np.min(window_times_us)
        t_max_spill_ms = (np.max(window_times_us) - t_start_spill_us) / 1000
        t_end_spill_ms = t_max_spill_ms - 500.0
        spill_timing_map[spill_id] = (t_start_spill_us, t_end_spill_ms)

df_pion_events["t_start_spill_us"] = df_pion_events["spill_counter"].map(
    lambda s: spill_timing_map[s][0] if s in spill_timing_map else 0.0
)
df_pion_events["t_end_spill [ms]"] = df_pion_events["spill_counter"].map(
    lambda s: spill_timing_map[s][1] if s in spill_timing_map else 0.0
)

df_pion_events["t_pi [ms]"] = (
    (df_pion_events["window_time"] / 1000.0)
    - df_pion_events["t_start_spill_us"]
) / 1000.0
df_pion_events = df_pion_events.drop(columns=["t_start_spill_us"])

dt = df_pion_events["t_end_spill [ms]"] - df_pion_events["t_pi [ms]"]

# ==============================================================================
# PROBABILITIES (P(t))
# ==============================================================================
df_pion_events["p_12B_early"] = np.exp(-dt / TAU_12B) * (
    np.exp(-t_start_win / TAU_12B) - np.exp(-t_threshold / TAU_12B)
)
df_pion_events["p_9Li_early"] = np.exp(-dt / TAU_9LI) * (
    np.exp(-t_start_win / TAU_9LI) - np.exp(-t_threshold / TAU_9LI)
)
df_pion_events["p_16N_early"] = np.exp(-dt / TAU_16N) * (
    np.exp(-t_start_win / TAU_16N) - np.exp(-t_threshold / TAU_16N)
)

df_pion_events["p_12B_late"] = np.exp(-dt / TAU_12B) * (
    np.exp(-t_threshold / TAU_12B) - np.exp(-t_end_win / TAU_12B)
)
df_pion_events["p_9Li_late"] = np.exp(-dt / TAU_9LI) * (
    np.exp(-t_threshold / TAU_9LI) - np.exp(-t_end_win / TAU_9LI)
)
df_pion_events["p_16N_late"] = np.exp(-dt / TAU_16N) * (
    np.exp(-t_threshold / TAU_16N) - np.exp(-t_end_win / TAU_16N)
)

for col in [
    "p_12B_early",
    "p_9Li_early",
    "p_16N_early",
    "p_12B_late",
    "p_9Li_late",
    "p_16N_late",
]:
    df_pion_events[col] = np.where(dt >= 0, df_pion_events[col], 0.0)

# ==============================================================================
# N_pi,scale AND N_exp COMPUTATION (SLIDE FORMULA IMPLEMENTATION)
# ==============================================================================
n_pions_total_filtered = float(df_pion_events["event_number"].count())
n_spills_filtered = float(df_pion_events["spill_counter"].nunique())

# Formula: N_pi,scale = N_pi * (N_spills,filtered / N_spills,total)
# Where both parameters are obtained post detector quality checking
n_pi_scale = n_pions_total_filtered * (n_spills_filtered / n_spills_total)

# Temporal mean probability of filtered pions <P(t)>
mean_p_12B_early = df_pion_events["p_12B_early"].mean()
mean_p_9Li_early = df_pion_events["p_9Li_early"].mean()
mean_p_16N_early = df_pion_events["p_16N_early"].mean()

mean_p_12B_late = df_pion_events["p_12B_late"].mean()
mean_p_9Li_late = df_pion_events["p_9Li_late"].mean()
mean_p_16N_late = df_pion_events["p_16N_late"].mean()

# N_exp = N_pi,scale * <P(t)>
n_12b_exp_early = n_pi_scale * mean_p_12B_early
n_9li_exp_early = n_pi_scale * mean_p_9Li_early
n_16n_exp_early = n_pi_scale * mean_p_16N_early

n_12b_exp_late = n_pi_scale * mean_p_12B_late
n_9li_exp_late = n_pi_scale * mean_p_9Li_late
n_16n_exp_late = n_pi_scale * mean_p_16N_late

# ==============================================================================
# RUN-LEVEL AGGREGATION AND OUTPUT WRITING
# ==============================================================================
lbl_12B_early = f"N 12B exp ({t_start_win:.0f}-{t_threshold:.0f} ms)"
lbl_9Li_early = f"N Li9 exp ({t_start_win:.0f}-{t_threshold:.0f} ms)"
lbl_16N_early = f"N 16N exp ({t_start_win:.0f}-{t_threshold:.0f} ms)"
lbl_12B_late = f"N 12B exp ({t_threshold:.0f}-{t_end_win:.0f} ms)"
lbl_9Li_late = f"N Li9 exp ({t_threshold:.0f}-{t_end_win:.0f} ms)"
lbl_16N_late = f"N 16N exp ({t_threshold:.0f}-{t_end_win:.0f} ms)"

df_by_run = pd.DataFrame(
    {
        "Run": [run_number],
        "Beam p (MeV/c)": [beam_momentum],
        "N spills total": [n_spills_total],
        "N spills with pions": [n_spills_filtered],
        "N pions (filtered)": [n_pions_total_filtered],
        "N pi scale": [round(n_pi_scale, 2)],
        lbl_12B_early: [round(n_12b_exp_early, 2)],
        lbl_9Li_early: [round(n_9li_exp_early, 2)],
        lbl_16N_early: [round(n_16n_exp_early, 2)],
        lbl_12B_late: [round(n_12b_exp_late, 2)],
        lbl_9Li_late: [round(n_9li_exp_late, 2)],
        lbl_16N_late: [round(n_16n_exp_late, 2)],
    }
)

# Save using the run number in the CSV filename
out_csv_path = os.path.join(outdir, f"summary_R{run_number}.csv")
df_by_run.to_csv(out_csv_path, index=False)
print(f" [SUCCESS] File saved at: {out_csv_path}\n")