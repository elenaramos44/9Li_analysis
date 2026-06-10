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
# CONTROLLER DE ARGUMENTOS DE LÍNEA DE COMANDOS
# ==============================================================================
parser = argparse.ArgumentParser(
    description="Procesa un archivo ROOT de WCTE para calcular producción de isótopos a nivel de Run."
)
parser.add_argument(
    "--input", required=True, type=str, help="Ruta al archivo .root a procesar"
)
parser.add_argument(
    "--outdir", required=True, type=str, help="Directorio donde guardar el .csv"
)
args = parser.parse_args()

filename = args.input
outdir = args.outdir

os.makedirs(outdir, exist_ok=True)

# ==============================================================================
# CONFIGURACIÓN Y PARÁMETROS FÍSICOS (DETECCIÓN DINÁMICA)
# ==============================================================================
# Detectar el beam momentum basado en la carpeta contenedora (p_340 -> -340, p_260 -> -260)
beam_momentum = -340 # Valor por defecto por si acaso
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

# Extraer el número de Run dinámicamente del nombre del archivo
match = re.search(r"R(\d{4})\.root", os.path.basename(filename))
run_number = match.group(1) if match else "Unknown"

print(f">>> Iniciando Run {run_number} | Beam p: {beam_momentum} MeV/c...")

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
# FILTERING and PION SELECTION
# ==============================================================================
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

eveto_cut_value = arrays_scalars["act_tagger_cut"][0]
pion_mask = filtered_windows["vme_act_tagger"] < eveto_cut_value
pion_events = filtered_windows[pion_mask]

if len(pion_events) == 0:
    print(
        f" [AVISO] Cero piones detectados en Run {run_number}. Finalizando de forma limpia."
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
# ALINEACIÓN TEMPORAL DE SPILLS
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
# PROBABILITIES
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
# AGREGACIÓN POR RUN Y ESCRITURA DE OUTPUTS
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
        "N spills with pions": [df_pion_events["spill_counter"].nunique()],
        "N pions (total)": [df_pion_events["event_number"].count()],
        lbl_12B_early: [df_pion_events["p_12B_early"].sum().round(2)],
        lbl_9Li_early: [df_pion_events["p_9Li_early"].sum().round(2)],
        lbl_16N_early: [df_pion_events["p_16N_early"].sum().round(2)],
        lbl_12B_late: [df_pion_events["p_12B_late"].sum().round(2)],
        lbl_9Li_late: [df_pion_events["p_9Li_late"].sum().round(2)],
        lbl_16N_late: [df_pion_events["p_16N_late"].sum().round(2)],
    }
)

# Guardar usando el número de run en el nombre del CSV
out_csv_path = os.path.join(outdir, f"summary_R{run_number}.csv")
df_by_run.to_csv(out_csv_path, index=False)
print(f" [ÉXITO] Archivo guardado en: {out_csv_path}\n")