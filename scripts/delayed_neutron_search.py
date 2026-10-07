#!/usr/bin/env python3

import os
import sys
import argparse
import pickle
import ast

import numpy as np
import pandas as pd
import awkward as ak
import uproot

sys.path.append("/scratch/elena/9Li")
import functions_multilateration


# =============================================================================
# Defaults
# =============================================================================

BASE_DIR = "/scratch/elena/9Li/results"

FV_TAG_DEFAULT = "FV_1"

PROMPT_WINDOW_NS_DEFAULT = 20.0
PROMPT_SKIP_MS_DEFAULT = 20.0

GAP_US_DEFAULT = 5.0
SEARCH_US_DEFAULT = 150.0

DELAYED_WINDOW_NS_DEFAULT = 10.0
DELAYED_NHITS_MIN_DEFAULT = 10
DELAYED_NHITS_MAX_DEFAULT = 50
DELAYED_RMS_CUT_NS_DEFAULT = 10.0

# Prompt-delayed spatial coincidence.
# 200 mm = 20 cm, following the AmBe selection scale.
DELAYED_DR_MAX_CM_DEFAULT = 20.0

# Stage-1 actually uses 480 ms:
# t_start = t_end - 0.48e9
LI9_TRIGGER_WINDOW_MS_DEFAULT = 480.0


# =============================================================================
# Delayed neutron sliding-window search
# =============================================================================

def find_delayed_candidates(
    times,
    window_ns,
    nhits_min,
    nhits_max,
    rms_cut_ns,
):
    """
    Search all hit-aligned sliding windows.

    A hit at times[i] is used as the start of a possible window:
        [times[i], times[i] + window_ns)

    No jump is made after finding a candidate, so no candidate window
    is missed.
    """

    if len(times) == 0:
        return []

    times = np.sort(np.asarray(times, dtype=float))
    candidates = []

    n = len(times)

    for i in range(n):

        t0 = times[i]

        j = np.searchsorted(
            times,
            t0 + window_ns,
            side="left",
        )

        nhits = j - i

        if nhits < nhits_min:
            continue

        if nhits > nhits_max:
            continue

        candidate_times = times[i:j]

        t_rms = float(np.std(candidate_times))

        if t_rms <= rms_cut_ns:
            candidates.append({
                "time": float(t0),
                "nhits": int(nhits),
                "t_rms": t_rms,
                "start_index": int(i),
                "stop_index": int(j),
            })

    return candidates


# =============================================================================
# Remove overlapping windows belonging to the same physical cluster
# =============================================================================

def deduplicate_candidates(candidates, window_ns):
    """
    Merge overlapping valid sliding-window candidates.

    Several 10 ns windows can describe the same physical light cluster.
    One representative is kept per overlapping group:

        highest nHits
        then lowest tRMS
        then earliest time
    """

    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda c: c["time"],
    )

    groups = []
    current_group = []
    current_end = -np.inf

    for candidate in candidates:

        start = candidate["time"]
        end = start + window_ns

        if not current_group or start < current_end:

            current_group.append(candidate)
            current_end = max(current_end, end)

        else:

            groups.append(current_group)
            current_group = [candidate]
            current_end = end

    if current_group:
        groups.append(current_group)

    output = []

    for group in groups:

        best = min(
            group,
            key=lambda c: (
                -c["nhits"],
                c["t_rms"],
                c["time"],
            ),
        )

        best = best.copy()

        best["n_overlapping_windows"] = len(group)

        output.append(best)

    return output


# =============================================================================
# Delayed candidate reconstruction
# =============================================================================

def reconstruct_delayed_candidate(times, mpmt_ids, pmt_ids):

    result = {
        "fit_success": False,
        "delayed_x": np.nan,
        "delayed_y": np.nan,
        "delayed_z": np.nan,
        "vertex_chi2_ndof": np.nan,
        "n_hits_used": np.nan,
    }

    if len(times) == 0:
        return result

    try:

        vertex = functions_multilateration.run_multilateration_candidate(
            times,
            mpmt_ids,
            pmt_ids,
            sigma_t=1.0,
            early_window_ns=100.0,
            robust_loss="soft_l1",
        )

        if vertex is None:
            return result

        result["fit_success"] = True

        for key, outkey in (
            ("x", "delayed_x"),
            ("y", "delayed_y"),
            ("z", "delayed_z"),
        ):
            if key in vertex and vertex[key] is not None:
                result[outkey] = float(
                    np.asarray(vertex[key]).reshape(-1)[0]
                )

        if vertex.get("chi2_ndof") is not None:
            result["vertex_chi2_ndof"] = float(
                np.asarray(vertex["chi2_ndof"]).reshape(-1)[0]
            )

        if vertex.get("n_hits_used") is not None:
            result["n_hits_used"] = float(
                np.asarray(vertex["n_hits_used"]).reshape(-1)[0]
            )

    except Exception as exc:
        result["fit_error"] = str(exc)

    return result


# =============================================================================
# ROOT loading
# =============================================================================

def load_root_chunk(root_file, entry_start, entry_stop):

    tree = root_file["WCTEReadoutWindows"]

    branches = [
        "window_time",
        "spill_counter",
        "hit_pmt_calibrated_times",
        "hit_mpmt_slot_ids",
        "hit_pmt_position_ids",
    ]

    arrays = tree.arrays(
        branches,
        entry_start=entry_start,
        entry_stop=entry_stop,
        library="ak",
    )

    window_time = np.asarray(
        arrays["window_time"],
        dtype=float,
    )

    spill_counter = np.asarray(
        arrays["spill_counter"],
        dtype=np.int64,
    )

    hit_times = arrays["hit_pmt_calibrated_times"]
    hit_mpmt = arrays["hit_mpmt_slot_ids"]
    hit_pmt = arrays["hit_pmt_position_ids"]

    counts = ak.to_numpy(
        ak.num(hit_times, axis=1)
    )

    if len(counts) == 0 or np.sum(counts) == 0:
        return {
            "time": np.array([], dtype=float),
            "spill": np.array([], dtype=np.int64),
            "mpmt": np.array([], dtype=np.int64),
            "pmt": np.array([], dtype=np.int64),
        }

    window_indices = np.repeat(
        np.arange(len(counts)),
        counts,
    )

    flat_hit_times = np.asarray(
        ak.flatten(hit_times, axis=None),
        dtype=float,
    )

    flat_mpmt = np.asarray(
        ak.flatten(hit_mpmt, axis=None),
        dtype=np.int64,
    )

    flat_pmt = np.asarray(
        ak.flatten(hit_pmt, axis=None),
        dtype=np.int64,
    )

    abs_hit_time = (
        window_time[window_indices]
        + flat_hit_times
    )

    hit_spill = spill_counter[window_indices]

    return {
        "time": abs_hit_time,
        "spill": hit_spill,
        "mpmt": flat_mpmt,
        "pmt": flat_pmt,
    }


# =============================================================================
# Chunk-map loading
# =============================================================================

def load_chunk(chunk_map_path, chunk_id):

    if not os.path.exists(chunk_map_path):
        raise FileNotFoundError(
            f"Chunk map not found:\n{chunk_map_path}"
        )

    if chunk_map_path.endswith(".csv"):

        chunk_map = pd.read_csv(chunk_map_path)

    elif chunk_map_path.endswith(".pkl"):

        with open(chunk_map_path, "rb") as f:
            chunk_map = pickle.load(f)

    else:
        raise ValueError(
            "Chunk map must be .csv or .pkl"
        )

    if isinstance(chunk_map, list):

        rows = [
            c for c in chunk_map
            if int(c["chunk_id"]) == chunk_id
        ]

        if not rows:
            raise ValueError(
                f"Chunk ID {chunk_id} not found in:\n"
                f"{chunk_map_path}"
            )

        return rows[0]

    if isinstance(chunk_map, pd.DataFrame):

        if "chunk_id" not in chunk_map.columns:
            raise ValueError(
                "Chunk map is missing 'chunk_id'."
            )

        rows = chunk_map[
            chunk_map["chunk_id"] == chunk_id
        ]

        if len(rows) != 1:
            raise ValueError(
                f"Expected one chunk with ID {chunk_id}, "
                f"found {len(rows)}."
            )

        return rows.iloc[0].to_dict()

    raise TypeError(
        f"Unsupported chunk-map type: {type(chunk_map)}"
    )


# =============================================================================
# Main
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Stage-6 delayed neutron coincidence search "
            "using Stage-5 Final*.pkl prompt candidates."
        )
    )

    # -------------------------------------------------------------------------
    # Input
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--chunk-map",
        required=True,
    )

    parser.add_argument(
        "--chunk-id",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--bkg",
        action="store_true",
    )

    parser.add_argument(
        "--fvtag",
        default=FV_TAG_DEFAULT,
    )

    # -------------------------------------------------------------------------
    # Timing
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--prompt-window-ns",
        type=float,
        default=PROMPT_WINDOW_NS_DEFAULT,
    )

    parser.add_argument(
        "--prompt-skip-ms",
        type=float,
        default=PROMPT_SKIP_MS_DEFAULT,
    )

    parser.add_argument(
        "--trigger-window-ms",
        type=float,
        default=LI9_TRIGGER_WINDOW_MS_DEFAULT,
    )

    parser.add_argument(
        "--gap-us",
        type=float,
        default=GAP_US_DEFAULT,
    )

    parser.add_argument(
        "--search-us",
        type=float,
        default=SEARCH_US_DEFAULT,
    )

    # -------------------------------------------------------------------------
    # Delayed neutron cuts
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--window-ns",
        type=float,
        default=DELAYED_WINDOW_NS_DEFAULT,
    )

    parser.add_argument(
        "--nhits-min",
        type=int,
        default=DELAYED_NHITS_MIN_DEFAULT,
    )

    parser.add_argument(
        "--nhits-max",
        type=int,
        default=DELAYED_NHITS_MAX_DEFAULT,
    )

    parser.add_argument(
        "--rms-cut-ns",
        type=float,
        default=DELAYED_RMS_CUT_NS_DEFAULT,
    )

    parser.add_argument(
        "--dr-max-cm",
        type=float,
        default=DELAYED_DR_MAX_CM_DEFAULT,
    )

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--outtag",
        default="",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args()

    # =========================================================================
    # Timing in ns
    # =========================================================================

    trigger_window_ns = args.trigger_window_ms * 1e6
    prompt_skip_ns = args.prompt_skip_ms * 1e6
    gap_ns = args.gap_us * 1e3
    search_ns = args.search_us * 1e3

    max_prompt_rel_ns = (
        trigger_window_ns
        - gap_ns
        - search_ns
    )

    # =========================================================================
    # Configuration
    # =========================================================================

    print()
    print("=" * 80)
    print("Stage-6 delayed neutron search")
    print("=" * 80)

    print(f"Chunk map              : {args.chunk_map}")
    print(f"Chunk ID               : {args.chunk_id}")
    print(f"Background             : {args.bkg}")
    print(f"FV tag                 : {args.fvtag}")

    print()
    print("Timing:")
    print(f"  Li9 window           : {args.trigger_window_ms:.3f} ms")
    print(f"  Prompt skip          : {args.prompt_skip_ms:.3f} ms")
    print(f"  Prompt window        : {args.prompt_window_ns:.3f} ns")
    print(f"  Gap                  : {args.gap_us:.3f} us")
    print(f"  Delayed search       : {args.search_us:.3f} us")

    print()
    print("Delayed candidate cuts:")
    print(f"  Window               : {args.window_ns:.3f} ns")
    print(f"  nHits                : {args.nhits_min} - {args.nhits_max}")
    print(f"  tRMS                 : {args.rms_cut_ns:.3f} ns")
    print(f"  Prompt-delayed dR    : < {args.dr_max_cm:.1f} cm")

    print()
    print(
        "Maximum prompt relative time for full search: "
        f"{max_prompt_rel_ns / 1e6:.6f} ms"
    )

    print("=" * 80)
    print()

    # =========================================================================
    # Load chunk
    # =========================================================================

    chunk = load_chunk(
        args.chunk_map,
        args.chunk_id,
    )

    run = int(chunk["run"])
    file_path = str(chunk["file_path"])
    entry_start = int(chunk["entry_start"])
    entry_stop = int(chunk["entry_stop"])

    spill_ids = chunk["spill_ids"]

    if isinstance(spill_ids, str):

        try:
            spill_ids = ast.literal_eval(spill_ids)
        except Exception:
            spill_ids = [
                int(x)
                for x in spill_ids.split(",")
                if x.strip()
            ]

    spill_ids = set(
        int(x)
        for x in spill_ids
    )

    print(f"Run                    : {run}")
    print(f"ROOT file              : {file_path}")
    print(
        f"Entry range            : "
        f"{entry_start} - {entry_stop}"
    )
    print(
        f"Number of chunk spills : "
        f"{len(spill_ids)}"
    )
    print()

    # =========================================================================
    # Stage-5 Final file
    # =========================================================================

    fv_dir = os.path.join(
        BASE_DIR,
        f"run{run}",
        "processed",
        args.fvtag,
    )

    final_name = (
        f"Final_FV_Li9_clusters_run{run}_BKG.pkl"
        if args.bkg
        else f"Final_FV_Li9_clusters_run{run}.pkl"
    )

    final_file = os.path.join(
        fv_dir,
        final_name,
    )

    if not os.path.exists(final_file):
        raise FileNotFoundError(
            f"Stage-5 Final file not found:\n{final_file}"
        )

    print("Loading Stage-5 prompts:")
    print(final_file)

    df_prompts = pd.read_pickle(final_file)

    print(
        f"Total Stage-5 candidates: "
        f"{len(df_prompts)}"
    )

    required_columns = {
        "spill_id",
        "t_window_start_ns",
        "t_window_start_rel_ns",
    }

    missing = (
        required_columns
        - set(df_prompts.columns)
    )

    if missing:
        raise ValueError(
            "Stage-5 Final file is missing:\n"
            f"{sorted(missing)}"
        )

    # =========================================================================
    # Keep prompts in this chunk
    # =========================================================================

    df_prompts = df_prompts[
        df_prompts["spill_id"].isin(spill_ids)
    ].copy()

    print(
        f"Stage-5 candidates in chunk: "
        f"{len(df_prompts)}"
    )

    if len(df_prompts) == 0:
        print("No Stage-5 prompts belong to this chunk.")
        return

    # =========================================================================
    # Read raw ROOT hits
    # =========================================================================

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"ROOT file not found:\n{file_path}"
        )

    print()
    print("Reading raw ROOT hits...")

    with uproot.open(file_path) as root_file:
        hits = load_root_chunk(
            root_file,
            entry_start,
            entry_stop,
        )

    hit_times = hits["time"]
    hit_spills = hits["spill"]
    hit_mpmt = hits["mpmt"]
    hit_pmt = hits["pmt"]

    print(
        f"Total raw hits in chunk: "
        f"{len(hit_times)}"
    )

    if len(hit_times):

        order = np.lexsort(
            (
                hit_times,
                hit_spills,
            )
        )

        hit_times = hit_times[order]
        hit_spills = hit_spills[order]
        hit_mpmt = hit_mpmt[order]
        hit_pmt = hit_pmt[order]

    # =========================================================================
    # Output
    # =========================================================================

    delayed_candidates_output = []
    prompt_summary_output = []
    diagnostic_rows = []

    n_prompt_total = 0
    n_prompt_too_early = 0
    n_prompt_too_late = 0
    n_prompt_searched = 0
    n_prompt_no_hits = 0
    n_prompt_with_neutron = 0

    n_delayed_windows = 0
    n_delayed_clusters = 0
    n_delayed_spatial = 0

    # =========================================================================
    # Spill-by-spill
    # =========================================================================

    for spill_id, prompt_group in df_prompts.groupby(
        "spill_id",
        sort=True,
    ):

        spill_id = int(spill_id)

        n_prompt_total += len(prompt_group)

        # ---------------------------------------------------------------------
        # Raw hits belonging to this spill
        # ---------------------------------------------------------------------

        left = np.searchsorted(
            hit_spills,
            spill_id,
            side="left",
        )

        right = np.searchsorted(
            hit_spills,
            spill_id,
            side="right",
        )

        if right <= left:

            n_prompt_no_hits += len(prompt_group)

            if args.verbose:
                print(
                    f"Spill {spill_id}: no raw hits."
                )

            continue

        spill_times = hit_times[left:right]
        spill_mpmt = hit_mpmt[left:right]
        spill_pmt = hit_pmt[left:right]

        # ---------------------------------------------------------------------
        # Prompts
        # ---------------------------------------------------------------------

        for prompt_index, prompt_row in prompt_group.iterrows():

            prompt_abs_ns = float(
                prompt_row["t_window_start_ns"]
            )

            prompt_rel_ns = float(
                prompt_row["t_window_start_rel_ns"]
            )

            # ================================================================
            # Prompt skip
            # ================================================================

            if prompt_rel_ns < prompt_skip_ns:

                n_prompt_too_early += 1

                if args.verbose:
                    print(
                        f"Spill {spill_id}, prompt "
                        f"{prompt_index}: too early "
                        f"({prompt_rel_ns / 1e6:.3f} ms rel.)"
                    )

                continue

            # ================================================================
            # Require complete delayed search inside Li9 window
            # ================================================================

            if (
                prompt_rel_ns
                + gap_ns
                + search_ns
                > trigger_window_ns
            ):

                n_prompt_too_late += 1

                if args.verbose:
                    print(
                        f"Spill {spill_id}, prompt "
                        f"{prompt_index}: too late "
                        f"({prompt_rel_ns / 1e6:.6f} ms rel.)"
                    )

                continue

            n_prompt_searched += 1

            # ================================================================
            # Absolute search interval
            # ================================================================

            search_start_ns = (
                prompt_abs_ns
                + gap_ns
            )

            search_stop_ns = (
                search_start_ns
                + search_ns
            )

            # ================================================================
            # Raw hits in delayed-search interval
            # ================================================================

            search_left = np.searchsorted(
                spill_times,
                search_start_ns,
                side="left",
            )

            search_right = np.searchsorted(
                spill_times,
                search_stop_ns,
                side="left",
            )

            search_times = spill_times[
                search_left:search_right
            ]

            search_mpmt = spill_mpmt[
                search_left:search_right
            ]

            search_pmt = spill_pmt[
                search_left:search_right
            ]

            # ================================================================
            # Coverage diagnostic
            # ================================================================

            if len(search_times):

                coverage_us = float(
                    (
                        np.max(search_times)
                        - search_start_ns
                    ) / 1e3
                )

            else:

                coverage_us = 0.0

            # ================================================================
            # Sliding delayed-neutron search
            # ================================================================

            raw_candidates = find_delayed_candidates(
                search_times,
                window_ns=args.window_ns,
                nhits_min=args.nhits_min,
                nhits_max=args.nhits_max,
                rms_cut_ns=args.rms_cut_ns,
            )

            n_delayed_windows += len(raw_candidates)

            # Remove overlapping windows belonging to the same pulse.
            candidates = deduplicate_candidates(
                raw_candidates,
                window_ns=args.window_ns,
            )

            n_delayed_clusters += len(candidates)

            # ================================================================
            # Prompt vertex
            # ================================================================

            def get_prompt_vertex(fine, coarse):

                if fine in prompt_row.index:
                    value = prompt_row[fine]
                elif coarse in prompt_row.index:
                    value = prompt_row[coarse]
                else:
                    return np.nan

                try:
                    return float(value)
                except Exception:
                    return np.nan

            prompt_x = get_prompt_vertex(
                "v_x_fine",
                "v_x",
            )

            prompt_y = get_prompt_vertex(
                "v_y_fine",
                "v_y",
            )

            prompt_z = get_prompt_vertex(
                "v_z_fine",
                "v_z",
            )

            # ================================================================
            # Reconstruct candidates and apply spatial coincidence
            # ================================================================

            spatial_candidates = []

            for candidate_id, candidate in enumerate(candidates):

                delayed_time = candidate["time"]

                delayed_left = np.searchsorted(
                    search_times,
                    delayed_time,
                    side="left",
                )

                delayed_right = np.searchsorted(
                    search_times,
                    delayed_time + args.window_ns,
                    side="left",
                )

                times_delayed = search_times[
                    delayed_left:delayed_right
                ]

                mpmt_delayed = search_mpmt[
                    delayed_left:delayed_right
                ]

                pmt_delayed = search_pmt[
                    delayed_left:delayed_right
                ]

                reco = reconstruct_delayed_candidate(
                    times_delayed,
                    mpmt_delayed,
                    pmt_delayed,
                )

                dR = np.nan

                if all(
                    np.isfinite(x)
                    for x in (
                        prompt_x,
                        prompt_y,
                        prompt_z,
                        reco["delayed_x"],
                        reco["delayed_y"],
                        reco["delayed_z"],
                    )
                ):

                    dR = float(
                        np.sqrt(
                            (reco["delayed_x"] - prompt_x) ** 2
                            + (reco["delayed_y"] - prompt_y) ** 2
                            + (reco["delayed_z"] - prompt_z) ** 2
                        )
                    )

                # Spatial coincidence with the prompt Li9 vertex. Guardar SIEMPRE, pase o no el corte.

                diagnostic_rows.append({
                    "run": run,
                    "spill_id": spill_id,
                    "prompt_index": prompt_index,
                    "candidate_id": candidate_id,
                    "dR_cm": dR,
                    "fit_success": reco["fit_success"],
                    "delayed_nhits": candidate["nhits"],
                    "delayed_t_rms_ns": candidate["t_rms"],
                    "vertex_chi2_ndof": reco["vertex_chi2_ndof"],
                    "n_hits_used": reco["n_hits_used"],
                    "delta_t_us": (candidate["time"] - prompt_abs_ns) / 1e3,
                    "prompt_x": prompt_x, "prompt_y": prompt_y, "prompt_z": prompt_z,
                    "delayed_x": reco["delayed_x"],
                    "delayed_y": reco["delayed_y"],
                    "delayed_z": reco["delayed_z"],
                })


                # Quality cut on delayed vertex reconstruction
                if not reco["fit_success"]:
                    continue  

                if not np.isfinite(reco["vertex_chi2_ndof"]):
                    continue

                if reco["vertex_chi2_ndof"] >= 3:
                    continue

                # Spatial coincidence with the prompt
                dR = np.sqrt((reco["delayed_x"] - prompt_x) ** 2 + (reco["delayed_y"] - prompt_y) ** 2 + (reco["delayed_z"] - prompt_z) ** 2)


                if not np.isfinite(dR):
                    continue

                if dR >= args.dr_max_cm:
                    continue

                spatial_candidates.append({
                    "candidate_id": candidate_id,
                    "candidate": candidate,
                    "reco": reco,
                    "dR": dR,
                    "delayed_left": delayed_left,
                    "delayed_right": delayed_right,
                })

            n_delayed_spatial += len(spatial_candidates)

            # ================================================================
            # If several delayed candidates survive, keep the closest one
            # ================================================================

            if spatial_candidates:

                selected = min(
                    spatial_candidates,
                    key=lambda c: c["dR"],
                )

                selected_candidates = [selected]

                n_prompt_with_neutron += 1

            else:

                selected_candidates = []

            # ================================================================
            # Prompt summary
            # ================================================================

            prompt_summary = {
                "run": run,
                "spill_id": spill_id,
                "prompt_index": prompt_index,

                "prompt_time_ns": prompt_abs_ns,
                "prompt_time_rel_ns": prompt_rel_ns,

                "prompt_t_ms": prompt_abs_ns / 1e6,
                "prompt_t_rel_ms": prompt_rel_ns / 1e6,

                "prompt_end_ns": (
                    prompt_abs_ns
                    + args.prompt_window_ns
                ),

                "search_start_ns": search_start_ns,
                "search_stop_ns": search_stop_ns,

                "search_start_ms": search_start_ns / 1e6,
                "search_stop_ms": search_stop_ns / 1e6,

                "gap_us": args.gap_us,
                "search_us": args.search_us,

                "coverage_us": coverage_us,
                "n_raw_hits_in_search": len(search_times),

                "n_delayed_windows": len(raw_candidates),
                "n_delayed_clusters": len(candidates),
                "n_delayed_spatial": len(spatial_candidates),

                "n_delayed_candidates": len(selected_candidates),
                "has_delayed_neutron": bool(selected_candidates),

                "prompt_x": prompt_x,
                "prompt_y": prompt_y,
                "prompt_z": prompt_z,
            }

            # Copy Stage-5 information
            for column in (
                "nhits",
                "t_window_start_ns",
                "t_window_start_rel_ns",
                "t_window_end_ns",
                "t_mean",
                "t_rms",
                "v_x",
                "v_y",
                "v_z",
                "v_x_fine",
                "v_y_fine",
                "v_z_fine",
                "fit_success",
                "chi2_ndof",
            ):

                if column in prompt_row.index:

                    prompt_summary[
                        f"prompt_{column}"
                    ] = prompt_row[column]

            prompt_summary_output.append(
                prompt_summary
            )

            # ================================================================
            # No selected delayed candidate
            # ================================================================

            if not selected_candidates:
                continue

            # ================================================================
            # Store selected delayed candidate
            # ================================================================

            selected = selected_candidates[0]

            candidate = selected["candidate"]
            reco = selected["reco"]
            dR = selected["dR"]

            delayed_time = candidate["time"]

            delayed_output = {
                "run": run,
                "spill_id": spill_id,
                "prompt_index": prompt_index,
                "candidate_id": selected["candidate_id"],

                "prompt_time_ns": prompt_abs_ns,
                "prompt_time_rel_ns": prompt_rel_ns,

                "prompt_t_ms": prompt_abs_ns / 1e6,
                "prompt_t_rel_ms": prompt_rel_ns / 1e6,

                "prompt_end_ns": (
                    prompt_abs_ns
                    + args.prompt_window_ns
                ),

                "prompt_x": prompt_x,
                "prompt_y": prompt_y,
                "prompt_z": prompt_z,

                "search_start_ns": search_start_ns,
                "search_stop_ns": search_stop_ns,

                "search_start_ms": search_start_ns / 1e6,
                "search_stop_ms": search_stop_ns / 1e6,

                "gap_us": args.gap_us,
                "search_us": args.search_us,

                # This is the quantity to use for the neutron-capture
                # time distribution / fit.
                "delta_t_us": (
                    delayed_time - prompt_abs_ns
                ) / 1e3,

                "delayed_time_ns": delayed_time,
                "delayed_t_ms": delayed_time / 1e6,

                "delayed_nhits": candidate["nhits"],
                "delayed_t_rms_ns": candidate["t_rms"],

                "n_overlapping_windows": (
                    candidate["n_overlapping_windows"]
                ),

                "delayed_x": reco["delayed_x"],
                "delayed_y": reco["delayed_y"],
                "delayed_z": reco["delayed_z"],

                "vertex_chi2_ndof": (
                    reco["vertex_chi2_ndof"]
                ),

                "fit_success": reco["fit_success"],
                "n_hits_used": reco["n_hits_used"],

                "dR_cm": dR,

                "coverage_us": coverage_us,
                "n_raw_hits_in_search": len(search_times),
            }

            if "fit_error" in reco:
                delayed_output["fit_error"] = reco["fit_error"]

            delayed_candidates_output.append(
                delayed_output
            )

    # =========================================================================
    # DataFrames
    # =========================================================================

    df_delayed = pd.DataFrame(
        delayed_candidates_output
    )

    df_prompt_summary = pd.DataFrame(
        prompt_summary_output
    )

    df_diagnostic = pd.DataFrame(diagnostic_rows)

    # =========================================================================
    # Output
    # =========================================================================

    output_dir = os.path.join(
        fv_dir,
        "delayed",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    bkg_tag = "_BKG" if args.bkg else ""

    candidate_output_path = os.path.join(
        output_dir,
        (
            f"Delayed_Li9_candidates_chunk_"
            f"{args.chunk_id}"
            f"{bkg_tag}"
            f"{args.outtag}.pkl"
        ),
    )

    prompt_output_path = os.path.join(
        output_dir,
        (
            f"Delayed_Li9_prompts_chunk_"
            f"{args.chunk_id}"
            f"{bkg_tag}"
            f"{args.outtag}.pkl"
        ),
    )

    df_delayed.to_pickle(
        candidate_output_path
    )

    df_prompt_summary.to_pickle(
        prompt_output_path
    )

    diagnostic_output_path = os.path.join(
        output_dir,
        f"Delayed_Li9_diagnostic_chunk_{args.chunk_id}{bkg_tag}{args.outtag}.pkl",
    )

    df_diagnostic.to_pickle(diagnostic_output_path)
    print(f"\nDiagnostic output: {diagnostic_output_path}")



    # =========================================================================
    # Summary
    # =========================================================================

    print()
    print("=" * 80)
    print("Stage-6 summary")
    print("=" * 80)

    print(
        f"Stage-5 prompts in chunk          : "
        f"{n_prompt_total}"
    )

    print(
        f"Prompts before "
        f"{args.prompt_skip_ms:g} ms               : "
        f"{n_prompt_too_early}"
    )

    print(
        f"Prompts too late for full search : "
        f"{n_prompt_too_late}"
    )

    print(
        f"Prompts actually searched        : "
        f"{n_prompt_searched}"
    )

    print(
        f"Prompts with no raw hits         : "
        f"{n_prompt_no_hits}"
    )

    print(
        f"Valid {args.window_ns:g} ns sliding windows      : "
        f"{n_delayed_windows}"
    )

    print(
        f"Non-overlapping delayed clusters : "
        f"{n_delayed_clusters}"
    )

    print(
        f"Clusters with dR < "
        f"{args.dr_max_cm:g} cm              : "
        f"{n_delayed_spatial}"
    )

    print(
        f"Prompts with selected neutron    : "
        f"{n_prompt_with_neutron}"
    )

    print()
    print("Delayed candidates output:")
    print(candidate_output_path)

    print()
    print("Prompt summary output:")
    print(prompt_output_path)

    print("=" * 80)
    print()


if __name__ == "__main__":
    main()