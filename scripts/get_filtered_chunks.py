#!/usr/bin/env python3
import os
import math
import glob
import re
import uproot
import argparse

def main():
    parser = argparse.ArgumentParser(description="Calculate chunks for signal or background.")
    parser.add_argument("--bkg", action="store_true", help="Process background files instead of signal")
    args = parser.parse_args()

    base_dir = "/scratch/elena/9Li/filtered_root"
    chunk_size = 25000
    
    # Seleccionar el sufijo dinámicamente según el argumento --bkg
    suffix = "_bkg.root" if args.bkg else "_signal.root"
    mode_str = "BACKGROUND" if args.bkg else "SIGNAL"
    
    search_path = os.path.join(base_dir, "*", f"*{suffix}")
    files = sorted(glob.glob(search_path))
    
    if not files:
        print(f"No se encontraron archivos con sufijo {suffix} en {base_dir}")
        return

    runs = []
    paths = []
    chunks_per_run = []
    total_global_chunks = 0

    print(f"=== RE-EVALUACIÓN DE CHUNKS ({mode_str}) ===")
    print(f"{'Run':<6} | {'Ventanas':<10} | {'Chunks':<6} | {'Ruta Relativa'}")
    print("-" * 60)

    # Evitamos la barra invertida dentro de la f-string definiendo el sufijo escapado antes
    escaped_suffix = suffix.replace(".", "\\.")
    pattern = rf'R(\d+){escaped_suffix}'

    for f_path in files:
        match = re.search(pattern, os.path.basename(f_path))
        if not match:
            continue
        run_num = int(match.group(1))
        
        momentum_dir = os.path.basename(os.path.dirname(f_path))
        
        try:
            with uproot.open(f_path) as f:
                tree = f["WCTEReadoutWindows"]
                total_events = tree.num_entries
                
            num_chunks = math.ceil(total_events / chunk_size)
            
            runs.append(run_num)
            paths.append(f'"{os.path.dirname(f_path)}"')
            chunks_per_run.append(num_chunks)
            total_global_chunks += num_chunks
            
            print(f"{run_num:<6} | {total_events:<10} | {num_chunks:<6} | {momentum_dir}")
            
        except Exception as e:
            print(f"Error leyendo {os.path.basename(f_path)}: {e}")

    print("\n" + "="*60)
    print("COPIAR Y PEGAR EN BASH (.sh)")
    print("="*60 + "\n")
    
    if total_global_chunks > 0:
        print(f"#SBATCH --array=0-{total_global_chunks - 1}%50\n")
    else:
        print("#SBATCH --array=0-0%50\n")
        
    print(f"RUNS=({' '.join(map(str, runs))})")
    print("\nPATHS=(")
    for p in paths:
        print(f"    {p}")
    print(")")
    print(f"\nCHUNKS_PER_RUN=({' '.join(map(str, chunks_per_run))})")

if __name__ == "__main__":
    main()