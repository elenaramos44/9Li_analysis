#!/usr/bin/env python3
import os
import math
import glob
import re
import uproot

def main():
    # Rutas base de tus archivos ya filtrados
    base_dir = "/scratch/elena/9Li/filtered_root"
    chunk_size = 25000
    
    # Buscamos todos los archivos de señal (cambia a '_bkg.root' cuando analices el fondo)
    search_path = os.path.join(base_dir, "*", "*_bkg.root")
    files = sorted(glob.glob(search_path))
    
    if not files:
        print(f"No se encontraron archivos filtrados en {base_dir}")
        return

    runs = []
    paths = []
    chunks_per_run = []
    total_global_chunks = 0

    print("=== RE-EVALUACIÓN DE CHUNKS (MUESTRA DE SEÑAL) ===")
    print(f"{'Run':<6} | {'Ventanas':<10} | {'Chunks':<6} | {'Ruta Relativa'}")
    print("-" * 60)

    for f_path in files:
        # Extraer el número de run del nombre del archivo
        match = re.search(r'R(\d+)_bkg\.root', os.path.basename(f_path))
        if not match:
            continue
        run_num = int(match.group(1))
        
        # Extraer la carpeta contenedora (p_340 o p_260)
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
    print("📋 COPIA Y PEGA ESTOS BLOQUES EN TU SCRIPT DE BASH (.sh)")
    print("="*60 + "\n")
    
    print(f"#SBATCH --array=0-{total_global_chunks - 1}%50\n")
    
    print(f"RUNS=({' '.join(map(str, runs))})")
    print("\nPATHS=(")
    for p in paths:
        print(f"    {p}")
    print(")")
    print(f"\nCHUNKS_PER_RUN=({' '.join(map(str, chunks_per_run))})")

if __name__ == "__main__":
    main()