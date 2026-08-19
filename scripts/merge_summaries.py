import glob
import os
import pandas as pd

# Directory where your individual CSV files are located
output_dir = "/scratch/elena/9Li/results/isotopes_output"

# Find all summary_R*.csv files
all_files = glob.glob(os.path.join(output_dir, "summary_R*.csv"))

if not all_files:
    print("No summary_R*.csv files found to merge.")
else:
    # Read and concatenate all dataframes
    df_list = [pd.read_csv(f) for f in all_files]
    master_df = pd.concat(df_list, ignore_index=True)

    # Sort by Run number
    master_df["Run"] = master_df["Run"].astype(int)
    master_df = master_df.sort_values("Run").reset_index(drop=True)

    # Save the updated master table
    master_path = os.path.join(output_dir, "master_table_all_runs.csv")
    master_df.to_csv(master_path, index=False)

    print(f"Success! Successfully merged {len(all_files)} runs.")
    print(f"Master table saved at: {master_path}")
    print("\nPreview of the resulting table:")
    print(master_df[["Run", "Beam p (MeV/c)", "N pions (filtered)"]])