#!/usr/bin/env python3

import os
import sys
import math
import pickle
import argparse

import numpy as np
import uproot


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a global spill-aware chunk map for the Li9 analysis. "
            "Each chunk contains complete spills and a spill_counter is "
            "never split between two chunks."
        )
    )

    parser.add_argument(
        "--base-root",
        type=str,
        required=True,
        help="Base directory containing p_260, p_340 and p_370"  #p_270 and p_350 para Gd runs
    )

    parser.add_argument(
        "--suffix",
        type=str,
        required=True,
        choices=["_signal.root", "_bkg.root"],
        help="ROOT sample suffix to process"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=25000,
        help="Target number of ROOT entries per chunk"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output pickle file containing the global chunk map"
    )

    return parser.parse_args()


def get_spill_ranges(tree):
    """
    Read spill_counter and determine contiguous entry ranges belonging
    to each spill.

    Returns
    -------
    list of dictionaries

        [
            {
                "spill_id": ...,
                "entry_start": ...,
                "entry_stop": ...,
                "n_entries": ...
            },
            ...
        ]

    entry_stop is exclusive.
    """

    spill_counter = tree["spill_counter"].array(library="np")

    if len(spill_counter) == 0:
        return []

    spills = np.asarray(spill_counter)

    spill_ranges = []

    current_spill = spills[0]
    start_entry = 0

    for i in range(1, len(spills)):

        if spills[i] != current_spill:

            spill_ranges.append(
                {
                    "spill_id": int(current_spill),
                    "entry_start": int(start_entry),
                    "entry_stop": int(i),
                    "n_entries": int(i - start_entry),
                }
            )

            current_spill = spills[i]
            start_entry = i

    # Last spill
    spill_ranges.append(
        {
            "spill_id": int(current_spill),
            "entry_start": int(start_entry),
            "entry_stop": int(len(spills)),
            "n_entries": int(len(spills) - start_entry),
        }
    )

    return spill_ranges


def build_chunks_for_file(
    file_path,
    run,
    momentum_dir,
    chunk_size
):
    """
    Build spill-aware chunks for one ROOT file.

    A chunk can contain many complete spills, but a single spill
    is never split across chunks.

    If an individual spill itself contains more than chunk_size
    entries, that spill necessarily becomes a chunk larger than
    chunk_size because preserving spill integrity has priority.
    """

    print("")
    print("---------------------------------------------------------------")
    print(f"Reading: {file_path}")
    print(f"Run: {run}")
    print(f"Momentum directory: {momentum_dir}")
    print("---------------------------------------------------------------")

    with uproot.open(file_path) as f:

        if "WCTEReadoutWindows" not in f:
            raise RuntimeError(
                f"WCTEReadoutWindows not found in {file_path}"
            )

        tree = f["WCTEReadoutWindows"]

        n_entries = tree.num_entries

        print(f"Total ROOT entries: {n_entries:,}")

        if n_entries == 0:
            print("File is empty.")
            return []

        if "spill_counter" not in tree.keys():
            raise RuntimeError(
                f"'spill_counter' branch not found in {file_path}"
            )

        spill_ranges = get_spill_ranges(tree)

    print(f"Number of spills: {len(spill_ranges):,}")

    chunks = []

    current_chunk_start = None
    current_chunk_stop = None
    current_chunk_entries = 0
    current_spills = []

    for spill in spill_ranges:

        spill_start = spill["entry_start"]
        spill_stop = spill["entry_stop"]
        spill_entries = spill["n_entries"]

        # -------------------------------------------------------
        # First spill in a chunk
        # -------------------------------------------------------

        if current_chunk_start is None:

            current_chunk_start = spill_start
            current_chunk_stop = spill_stop
            current_chunk_entries = spill_entries
            current_spills = [spill]

            continue

        # -------------------------------------------------------
        # Would adding this COMPLETE spill exceed target size?
        # -------------------------------------------------------

        proposed_entries = current_chunk_entries + spill_entries

        if proposed_entries > chunk_size:

            # Close current chunk BEFORE this spill.
            chunks.append(
                {
                    "run": int(run),
                    "momentum_dir": momentum_dir,
                    "file_path": file_path,

                    "entry_start": int(current_chunk_start),
                    "entry_stop": int(current_chunk_stop),

                    "n_entries": int(current_chunk_entries),

                    "spill_start": int(current_spills[0]["spill_id"]),
                    "spill_stop": int(current_spills[-1]["spill_id"]),

                    "spill_ids": [
                        int(x["spill_id"])
                        for x in current_spills
                    ],

                    "n_spills": int(len(current_spills)),
                }
            )

            # Start new chunk with the COMPLETE spill.
            current_chunk_start = spill_start
            current_chunk_stop = spill_stop
            current_chunk_entries = spill_entries
            current_spills = [spill]

        else:

            # Add complete spill to current chunk.
            current_chunk_stop = spill_stop
            current_chunk_entries += spill_entries
            current_spills.append(spill)

    # -----------------------------------------------------------
    # Store final chunk
    # -----------------------------------------------------------

    if current_chunk_start is not None:

        chunks.append(
            {
                "run": int(run),
                "momentum_dir": momentum_dir,
                "file_path": file_path,

                "entry_start": int(current_chunk_start),
                "entry_stop": int(current_chunk_stop),

                "n_entries": int(current_chunk_entries),

                "spill_start": int(current_spills[0]["spill_id"]),
                "spill_stop": int(current_spills[-1]["spill_id"]),

                "spill_ids": [
                    int(x["spill_id"])
                    for x in current_spills
                ],

                "n_spills": int(len(current_spills)),
            }
        )

    # -----------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------

    print(f"Created {len(chunks):,} spill-aware chunks.")

    if chunks:

        sizes = [x["n_entries"] for x in chunks]

        print(
            f"Chunk entries: "
            f"min={min(sizes):,}, "
            f"max={max(sizes):,}, "
            f"mean={np.mean(sizes):,.1f}"
        )

        oversized = [
            x for x in chunks
            if x["n_entries"] > chunk_size
        ]

        if oversized:

            print(
                f"WARNING: {len(oversized)} chunk(s) exceed "
                f"the target size of {chunk_size:,} entries."
            )

            print(
                "This happens because an individual spill is larger "
                "than the target size and cannot be split."
            )

    return chunks


def main():

    args = parse_args()

    base_root = os.path.abspath(args.base_root)
    suffix = args.suffix
    chunk_size = args.chunk_size
    output = os.path.abspath(args.output)

    if chunk_size <= 0:
        raise ValueError(
            f"Chunk size must be positive. Got {chunk_size}"
        )

    if not os.path.isdir(base_root):
        raise RuntimeError(
            f"Base ROOT directory does not exist: {base_root}"
        )

    # -----------------------------------------------------------
    # These are the three UPW momentum directories
    # -----------------------------------------------------------

    #subdirs = [          #UPW data
    #    "p_260",
    #    "p_340",
    #    "p_370",
    #]

    subdirs = [           #Gd data
        "p_270",
        "p_350",
    ]

    print("================================================================")
    print("Building spill-aware Li9 chunk map")
    print("================================================================")
    print(f"Base ROOT directory : {base_root}")
    print(f"Sample suffix       : {suffix}")
    print(f"Target chunk size   : {chunk_size:,}")
    print(f"Output map          : {output}")
    print("")
    print("Momentum directories:")
    for sub in subdirs:
        print(f"  - {sub}")
    print("================================================================")

    # -----------------------------------------------------------
    # Discover ROOT files
    # -----------------------------------------------------------

    files_to_process = []

    for momentum_dir in subdirs:

        directory = os.path.join(
            base_root,
            momentum_dir
        )

        if not os.path.isdir(directory):

            print(
                f"WARNING: Directory does not exist, skipping: "
                f"{directory}"
            )

            continue

        for filename in sorted(os.listdir(directory)):

            if not filename.startswith(
                "WCTE_merged_production_R"
            ):
                continue

            if not filename.endswith(suffix):
                continue

            # Extract run number safely
            run_string = filename[
                len("WCTE_merged_production_R"):
                -len(suffix)
            ]

            try:
                run = int(run_string)
            except ValueError:

                print(
                    f"WARNING: Could not extract run number from "
                    f"{filename}. Skipping."
                )

                continue

            file_path = os.path.join(
                directory,
                filename
            )

            files_to_process.append(
                (
                    momentum_dir,
                    run,
                    file_path
                )
            )

    # -----------------------------------------------------------
    # Sort deterministically
    # -----------------------------------------------------------

    files_to_process.sort(
        key=lambda x: (
            x[0],
            x[1]
        )
    )

    print("")
    print(
        f"Found {len(files_to_process)} "
        f"{suffix} ROOT files."
    )

    if len(files_to_process) == 0:

        raise RuntimeError(
            "No matching ROOT files were found."
        )

    # -----------------------------------------------------------
    # Build global chunk list
    # -----------------------------------------------------------

    all_chunks = []

    global_chunk_id = 0

    total_entries = 0
    total_spills = 0

    for momentum_dir, run, file_path in files_to_process:

        file_chunks = build_chunks_for_file(
            file_path=file_path,
            run=run,
            momentum_dir=momentum_dir,
            chunk_size=chunk_size
        )

        for chunk in file_chunks:

            # Add GLOBAL task ID.
            #
            # This is the SLURM_ARRAY_TASK_ID that
            # load_SW_all_files.sh will use.
            chunk["chunk_id"] = int(global_chunk_id)

            all_chunks.append(chunk)

            global_chunk_id += 1

        # Statistics
        total_entries += sum(
            x["n_entries"]
            for x in file_chunks
        )

        total_spills += sum(
            x["n_spills"]
            for x in file_chunks
        )

    # -----------------------------------------------------------
    # Validate chunk map
    # -----------------------------------------------------------

    print("")
    print("================================================================")
    print("Validating global chunk map")
    print("================================================================")

    previous_file = None
    previous_stop = None

    for chunk in all_chunks:

        file_path = chunk["file_path"]

        start = chunk["entry_start"]
        stop = chunk["entry_stop"]

        if stop <= start:

            raise RuntimeError(
                f"Invalid chunk {chunk['chunk_id']}: "
                f"entry_start={start}, entry_stop={stop}"
            )

        if chunk["n_entries"] != stop - start:

            raise RuntimeError(
                f"Invalid entry count in chunk "
                f"{chunk['chunk_id']}"
            )

        # Check continuity only within the same ROOT file.
        if file_path == previous_file:

            if start != previous_stop:

                raise RuntimeError(
                    "Chunk map has a gap or overlap:"
                    f"\n  file={file_path}"
                    f"\n  previous_stop={previous_stop}"
                    f"\n  current_start={start}"
                )

        previous_file = file_path
        previous_stop = stop

    print(
        f"Validated {len(all_chunks):,} chunks successfully."
    )

    # -----------------------------------------------------------
    # Save map
    # -----------------------------------------------------------

    output_dir = os.path.dirname(output)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output, "wb") as f:

        pickle.dump(
            all_chunks,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    # -----------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------

    print("")
    print("================================================================")
    print("Spill-aware chunk map completed successfully")
    print("================================================================")
    print(f"Sample suffix      : {suffix}")
    print(f"ROOT files         : {len(files_to_process):,}")
    print(f"Total entries      : {total_entries:,}")
    print(f"Total spill groups : {total_spills:,}")
    print(f"Total chunks       : {len(all_chunks):,}")
    print(f"Target chunk size  : {chunk_size:,}")
    print(f"Output             : {output}")
    print("================================================================")

    print("")
    print("First chunks:")

    for chunk in all_chunks[:5]:

        print(
            f"  Task {chunk['chunk_id']:5d} | "
            f"{chunk['momentum_dir']:5s} | "
            f"R{chunk['run']} | "
            f"entries {chunk['entry_start']:,}-"
            f"{chunk['entry_stop']:,} | "
            f"{chunk['n_entries']:,} entries | "
            f"{chunk['n_spills']} spills"
        )

    if len(all_chunks) > 5:

        print("  ...")

        for chunk in all_chunks[-5:]:

            print(
                f"  Task {chunk['chunk_id']:5d} | "
                f"{chunk['momentum_dir']:5s} | "
                f"R{chunk['run']} | "
                f"entries {chunk['entry_start']:,}-"
                f"{chunk['entry_stop']:,} | "
                f"{chunk['n_entries']:,} entries | "
                f"{chunk['n_spills']} spills"
            )

    print("")
    print("Done.")


if __name__ == "__main__":
    try:
        main()

    except Exception as e:

        print("")
        print("================================================================")
        print("ERROR")
        print("================================================================")
        print(str(e))
        print("================================================================")

        sys.exit(1)