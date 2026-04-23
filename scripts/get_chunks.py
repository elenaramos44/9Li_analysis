import uproot
import math
import argparse
import os

def calculate_chunks(file_path, chunk_size):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    try:
        # Open the file and access the WCSim tree
        with uproot.open(file_path) as file:
            # Common WCSim tree name is 'wcsimT'
            tree = file["WCTEReadoutWindows"]
            total_events = tree.num_entries
            
            # Calculate chunks (rounding up)
            num_chunks = math.ceil(total_events / chunk_size)
            max_array_index = num_chunks - 1

            print(f"File: {os.path.basename(file_path)}")
            print(f"Total events: {total_events}")
            print(f"Chunk size:   {chunk_size}")
            print(f"Number of chunks: {num_chunks}")
            print(f"SLURM setting: --array=0-{max_array_index}")
            
    except Exception as e:
        print(f"Error reading ROOT file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate SLURM array range for WCSim runs.")
    parser.add_argument("file", help="Path to the .root file")
    parser.add_argument("--size", type=int, default=25000, help="Events per chunk (default: 25000)")

    args = parser.parse_args()
    calculate_chunks(args.file, args.size)