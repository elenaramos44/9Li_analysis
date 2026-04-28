#!/usr/bin/env python3
import pandas as pd
import ast
from glob import glob
import os

input_files = glob("/scratch/elena/9Li/results/run1846/multilat_output/*_multilat_chi2.csv")

outdir = "/scratch/elena/9Li/results/run1848/processed"
os.makedirs(outdir, exist_ok=True)

for f in input_files:
    print(f"Processing {f}...")

    df = pd.read_csv(f)

    # Convertir las columnas de listas (importante para que el PKL sea útil)
    # Usamos get() o comprobamos si existen para evitar errores si algún chunk falló
    for col in ['hit_times_ns', 'hit_slot_ids', 'hit_channel_ids']:
        if col in df.columns:
             df[col] = df[col].apply(ast.literal_eval)

    # Guardar como PKL (Binary format)
    #cambiar de .csv a .pkl
    outname = os.path.basename(f).replace("_multilat_chi2.csv", ".pkl")
    df.to_pickle(os.path.join(outdir, outname))

print("¡Conversión terminada!")