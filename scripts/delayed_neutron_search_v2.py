#!/usr/bin/env python3
"""
Stage-6 delayed-neutron search (v2)

Changes with respect to v1
--------------------------
1. Prompts are searched only inside [--prompt-min-ms, --prompt-max-ms] of the
   selected time axis. By default, this is 100-500 ms in the Li9 trigger window.
2. Prompts must pass fit_success & chi2_ndof < --prompt-chi2-max (default 3).
3. Prompt isolation: reject BOTH Stage-5 prompt candidates in a spill when their
   prompt windows overlap or are separated by <= --iso-margin-us (default 0.2 us).
   The hit counts in the margins are retained as diagnostics only.
4. Delayed vertex quality cut is configurable (--delayed-chi2-max, default 3).
5. Burst veto: v1 rejected windows above nhits_max BEFORE the overlap merge, so
   the shoulders of large bursts survived with exactly nhits_max hits (the spike
   at nHits=50). Now the merge uses all windows and nhits_max is applied to the
   representative. --legacy-nhits-cut restores the v1 behaviour.
6. Every prompt is logged with its status (Delayed_Li9_status_chunk_*.pkl).

Time axes
---------
 stage1 : t_window_start_rel_ns of Stage 1 (zero = t_end - 480 ms, i.e. about
          20 ms AFTER the end of the spill). This is the axis of the time
          distribution plots.
 li9    : zero = t_end - --li9-window-ms (500 ms), i.e. the start of the Li9
          trigger window = end of the spill. li9 = stage1 + 20 ms.
"""

import os
import sys
import argparse
import pickle
import ast
from collections import Counter

import numpy as np
import pandas as pd
import awkward as ak
import uproot

sys.path.append("/scratch/elena/9Li")
import functions_multilateration


BASE_DIR = "/scratch/elena/9Li/results"
STAGE1_WINDOW_MS = 480.0   # Stage 1: t_start = t_end - 0.48e9


# =============================================================================
# Delayed neutron sliding-window search
# =============================================================================

def find_delayed_candidates(times, window_ns, nhits_min, rms_cut_ns, nhits_max=None):
    """
    All hit-aligned sliding windows [t_i, t_i + window_ns) with
    nhits >= nhits_min and tRMS <= rms_cut_ns.

    nhits_max=None -> no upper cut here (applied later, after the overlap merge).
    """
    times = np.sort(np.asarray(times, dtype=float))
    n = len(times)
    if n == 0:
        return []

    stops = np.searchsorted(times, times + window_ns, side="left")
    counts = stops - np.arange(n)

    ok = counts >= nhits_min
    if nhits_max is not None:
        ok &= counts <= nhits_max

    out = []
    for i in np.nonzero(ok)[0]:
        seg = times[i:stops[i]]
        t_rms = float(np.std(seg))
        if t_rms <= rms_cut_ns:
            out.append({
                "time": float(times[i]),
                "nhits": int(counts[i]),
                "t_rms": t_rms,
                "start_index": int(i),
                "stop_index": int(stops[i]),
            })
    return out


def deduplicate_candidates(candidates, window_ns):
    """
    Merge overlapping windows that describe the same light cluster.
    Representative: highest nHits, then lowest tRMS, then earliest time.
    """
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda c: c["time"])

    groups, current, current_end = [], [], -np.inf
    for c in candidates:
        start, end = c["time"], c["time"] + window_ns
        if not current or start < current_end:
            current.append(c)
            current_end = max(current_end, end)
        else:
            groups.append(current)
            current, current_end = [c], end
    if current:
        groups.append(current)

    out = []
    for g in groups:
        best = min(g, key=lambda c: (-c["nhits"], c["t_rms"], c["time"])).copy()
        best["n_overlapping_windows"] = len(g)
        out.append(best)
    return out


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
            times, mpmt_ids, pmt_ids,
            sigma_t=1.0,
            early_window_ns=100.0,
            robust_loss="soft_l1",
        )
        if vertex is None:
            return result

        result["fit_success"] = True
        for key, outkey in (("x", "delayed_x"), ("y", "delayed_y"), ("z", "delayed_z")):
            if key in vertex and vertex[key] is not None:
                result[outkey] = float(np.asarray(vertex[key]).reshape(-1)[0])
        if vertex.get("chi2_ndof") is not None:
            result["vertex_chi2_ndof"] = float(np.asarray(vertex["chi2_ndof"]).reshape(-1)[0])
        if vertex.get("n_hits_used") is not None:
            result["n_hits_used"] = float(np.asarray(vertex["n_hits_used"]).reshape(-1)[0])

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
    arrays = tree.arrays(branches, entry_start=entry_start, entry_stop=entry_stop, library="ak")

    window_time = np.asarray(arrays["window_time"], dtype=float)
    spill_counter = np.asarray(arrays["spill_counter"], dtype=np.int64)

    hit_times = arrays["hit_pmt_calibrated_times"]
    hit_mpmt = arrays["hit_mpmt_slot_ids"]
    hit_pmt = arrays["hit_pmt_position_ids"]

    counts = ak.to_numpy(ak.num(hit_times, axis=1))
    if len(counts) == 0 or np.sum(counts) == 0:
        return {
            "time": np.array([], dtype=float),
            "spill": np.array([], dtype=np.int64),
            "mpmt": np.array([], dtype=np.int64),
            "pmt": np.array([], dtype=np.int64),
        }

    window_indices = np.repeat(np.arange(len(counts)), counts)
    flat_hit_times = np.asarray(ak.flatten(hit_times, axis=None), dtype=float)
    flat_mpmt = np.asarray(ak.flatten(hit_mpmt, axis=None), dtype=np.int64)
    flat_pmt = np.asarray(ak.flatten(hit_pmt, axis=None), dtype=np.int64)

    return {
        "time": window_time[window_indices] + flat_hit_times,
        "spill": spill_counter[window_indices],
        "mpmt": flat_mpmt,
        "pmt": flat_pmt,
    }


def load_chunk(chunk_map_path, chunk_id):
    if not os.path.exists(chunk_map_path):
        raise FileNotFoundError(f"Chunk map not found:\n{chunk_map_path}")

    if chunk_map_path.endswith(".csv"):
        chunk_map = pd.read_csv(chunk_map_path)
    elif chunk_map_path.endswith(".pkl"):
        with open(chunk_map_path, "rb") as f:
            chunk_map = pickle.load(f)
    else:
        raise ValueError("Chunk map must be .csv or .pkl")

    if isinstance(chunk_map, list):
        rows = [c for c in chunk_map if int(c["chunk_id"]) == chunk_id]
        if not rows:
            raise ValueError(f"Chunk ID {chunk_id} not found in:\n{chunk_map_path}")
        return rows[0]

    if isinstance(chunk_map, pd.DataFrame):
        rows = chunk_map[chunk_map["chunk_id"] == chunk_id]
        if len(rows) != 1:
            raise ValueError(f"Expected one chunk with ID {chunk_id}, found {len(rows)}.")
        return rows.iloc[0].to_dict()

    raise TypeError(f"Unsupported chunk-map type: {type(chunk_map)}")


def get_prompt_vertex(row, fine, coarse):
    for name in (fine, coarse):
        if name in row.index:
            try:
                return float(row[name])
            except Exception:
                return np.nan
    return np.nan


# =============================================================================
# Arguments
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Stage-6 delayed neutron coincidence search (v2).")

    p.add_argument("--chunk-map", required=True)
    p.add_argument("--chunk-id", required=True, type=int)
    p.add_argument("--bkg", action="store_true")
    p.add_argument("--fvtag", default="FV_1")

    # --- which prompts are searched ---
    p.add_argument("--time-ref", choices=["stage1", "li9"], default="li9",
                   help="zero of the prompt-time axis (stage1 = axis of the time-distribution plots, "
                        "li9 = true start of the Li9 trigger window = spill end)")
    p.add_argument("--li9-window-ms", type=float, default=500.0)
    p.add_argument("--prompt-min-ms", type=float, default=100.0)
    p.add_argument("--prompt-max-ms", type=float, default=500.0)
    p.add_argument("--prompt-window-ns", type=float, default=20.0)
    p.add_argument("--prompt-chi2-max", type=float, default=3.0)

    # --- prompt isolation ---
    p.add_argument("--iso-margin-us", type=float, default=0.2,
                   help="reject both prompt candidates if their time windows overlap or are separated by <= this")

    # --- delayed search ---
    p.add_argument("--gap-us", type=float, default=5.0)
    p.add_argument("--search-us", type=float, default=150.0)
    p.add_argument("--window-ns", type=float, default=5.0)
    p.add_argument("--nhits-min", type=int, default=10)
    p.add_argument("--nhits-max", type=int, default=50)
    p.add_argument("--rms-cut-ns", type=float, default=10.0)
    p.add_argument("--delayed-chi2-max", type=float, default=float("inf"))
    p.add_argument("--dr-max-cm", type=float, default=20.0)
    p.add_argument("--legacy-nhits-cut", action="store_true",
                   help="v1 behaviour: apply nhits_max before the overlap merge")

    p.add_argument("--outtag", default="")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    trigger_end_ms = STAGE1_WINDOW_MS if args.time_ref == "stage1" else args.li9_window_ms
    prompt_min_ns = args.prompt_min_ms * 1e6
    gap_ns = args.gap_us * 1e3
    search_ns = args.search_us * 1e3
    iso_margin_ns = args.iso_margin_us * 1e3
    max_prompt_rel_ns = min(args.prompt_max_ms * 1e6, trigger_end_ms * 1e6 - gap_ns - search_ns)

    print()
    print("=" * 80)
    print("Stage-6 delayed neutron search (v2)")
    print("=" * 80)
    print(f"Chunk map / ID         : {args.chunk_map} / {args.chunk_id}")
    print(f"Background             : {args.bkg}   FV tag: {args.fvtag}")
    print()
    print(f"Prompt time axis       : {args.time_ref}"
          f"  ({'zero = t_end - 480 ms (~20 ms after spill end)' if args.time_ref == 'stage1' else f'zero = t_end - {args.li9_window_ms:g} ms (= spill end)'})")
    print(f"Prompts searched in    : {args.prompt_min_ms:g} - {max_prompt_rel_ns / 1e6:.3f} ms")
    print(f"Prompt quality         : fit_success & chi2/ndof < {args.prompt_chi2_max:g}")
    print(
        f"Prompt isolation       : reject both candidates if their windows "
        f"overlap or are separated by <= {args.iso_margin_us:g} us"
    )
    print(f"Isolation hit counts   : diagnostics only (not used as a selection cut)")
    print(f"Delayed search         : gap {args.gap_us:g} us, window {args.search_us:g} us")
    print(f"Delayed cluster        : {args.window_ns:g} ns, nHits {args.nhits_min}-{args.nhits_max}, "
          f"tRMS <= {args.rms_cut_ns:g} ns, burst veto: {not args.legacy_nhits_cut}")
    print(f"Delayed vertex         : chi2/ndof < {args.delayed_chi2_max:g}, dR < {args.dr_max_cm:g} cm")
    print("=" * 80)
    print()

    # ------------------------------------------------------------------
    # Chunk and Stage-5 prompts
    # ------------------------------------------------------------------
    chunk = load_chunk(args.chunk_map, args.chunk_id)
    run = int(chunk["run"])
    file_path = str(chunk["file_path"])
    entry_start, entry_stop = int(chunk["entry_start"]), int(chunk["entry_stop"])

    spill_ids = chunk["spill_ids"]
    if isinstance(spill_ids, str):
        try:
            spill_ids = ast.literal_eval(spill_ids)
        except Exception:
            spill_ids = [int(x) for x in spill_ids.split(",") if x.strip()]
    spill_ids = set(int(x) for x in spill_ids)

    print(f"Run {run} | ROOT {file_path}")
    print(f"Entries {entry_start}-{entry_stop} | spills in chunk: {len(spill_ids)}\n")

    fv_dir = os.path.join(BASE_DIR, f"run{run}", "processed", args.fvtag)
    final_name = f"Final_FV_Li9_clusters_run{run}_BKG.pkl" if args.bkg else f"Final_FV_Li9_clusters_run{run}.pkl"
    final_file = os.path.join(fv_dir, final_name)
    if not os.path.exists(final_file):
        raise FileNotFoundError(f"Stage-5 Final file not found:\n{final_file}")

    df_prompts = pd.read_pickle(final_file)
    print(f"Stage-5 file: {final_file}\nTotal Stage-5 candidates: {len(df_prompts)}")

    missing = {"spill_id", "t_window_start_ns", "t_window_start_rel_ns"} - set(df_prompts.columns)
    if missing:
        raise ValueError(f"Stage-5 Final file is missing: {sorted(missing)}")

    has_quality = {"fit_success", "chi2_ndof"}.issubset(df_prompts.columns)
    if not has_quality:
        print("WARNING: fit_success / chi2_ndof not in the Stage-5 file: prompt quality cut NOT applied.")

    df_prompts = df_prompts[df_prompts["spill_id"].isin(spill_ids)].copy()
    print(f"Stage-5 candidates in chunk: {len(df_prompts)}")
    if len(df_prompts) == 0:
        print("No Stage-5 prompts belong to this chunk.")
        return

    # ------------------------------------------------------------------
    # Raw hits
    # ------------------------------------------------------------------
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ROOT file not found:\n{file_path}")

    print("\nReading raw ROOT hits...")
    with uproot.open(file_path) as root_file:
        hits = load_root_chunk(root_file, entry_start, entry_stop)

    hit_times, hit_spills = hits["time"], hits["spill"]
    hit_mpmt, hit_pmt = hits["mpmt"], hits["pmt"]
    print(f"Total raw hits in chunk: {len(hit_times)}")

    if len(hit_times):
        order = np.lexsort((hit_times, hit_spills))
        hit_times, hit_spills = hit_times[order], hit_spills[order]
        hit_mpmt, hit_pmt = hit_mpmt[order], hit_pmt[order]

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    delayed_out, prompt_out, diag_rows, status_rows = [], [], [], []
    status_counter = Counter()
    spill_spans_ms = []
    iso_before_list, iso_after_list = [], []

    n_windows = n_clusters = n_burst_rejected = n_spatial = n_with_neutron = 0

    for spill_id, group in df_prompts.groupby("spill_id", sort=True):
        spill_id = int(spill_id)

        left = np.searchsorted(hit_spills, spill_id, side="left")
        right = np.searchsorted(hit_spills, spill_id, side="right")

        if right <= left:
            for pidx in group.index:
                status_rows.append({"run": run, "spill_id": spill_id, "prompt_index": pidx, "status": "no_raw_hits"})
            status_counter["no_raw_hits"] += len(group)
            continue

        spill_times = hit_times[left:right]
        spill_mpmt = hit_mpmt[left:right]
        spill_pmt = hit_pmt[left:right]

        t_first, t_end_spill = float(spill_times[0]), float(spill_times[-1])
        spill_spans_ms.append((t_end_spill - t_first) / 1e6)
        li9_start_ns = t_end_spill - args.li9_window_ms * 1e6

        # --------------------------------------------------------------
        # Prompt isolation: compare all Stage-5 candidate windows in this
        # spill. If two windows overlap or are separated by <= the configured
        # margin, reject BOTH candidates. Isolation is checked against all
        # Stage-5 candidates in the spill, regardless of their later time/fit
        # selection status.
        # --------------------------------------------------------------
        prompt_starts = group["t_window_start_ns"].to_numpy(dtype=float)

        if "t_window_end_ns" in group.columns:
            prompt_ends = pd.to_numeric(
                group["t_window_end_ns"], errors="coerce"
            ).to_numpy(dtype=float)
            invalid_ends = (
                ~np.isfinite(prompt_ends)
                | (prompt_ends < prompt_starts)
            )
            prompt_ends[invalid_ends] = (
                prompt_starts[invalid_ends] + args.prompt_window_ns
            )
        else:
            prompt_ends = prompt_starts + args.prompt_window_ns

        group_indices = list(group.index)
        isolation_bad_indices = set()
        isolation_neighbor_counts = {idx: 0 for idx in group_indices}

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if prompt_ends[i] < prompt_starts[j]:
                    separation_ns = prompt_starts[j] - prompt_ends[i]
                elif prompt_ends[j] < prompt_starts[i]:
                    separation_ns = prompt_starts[i] - prompt_ends[j]
                else:
                    # The two prompt windows overlap.
                    separation_ns = 0.0

                if separation_ns <= iso_margin_ns:
                    idx_i = group_indices[i]
                    idx_j = group_indices[j]
                    isolation_bad_indices.add(idx_i)
                    isolation_bad_indices.add(idx_j)
                    isolation_neighbor_counts[idx_i] += 1
                    isolation_neighbor_counts[idx_j] += 1

        for prompt_index, prompt_row in group.iterrows():

            prompt_abs_ns = float(prompt_row["t_window_start_ns"])
            rel_stage1_ns = float(prompt_row["t_window_start_rel_ns"])
            rel_li9_ns = prompt_abs_ns - li9_start_ns
            prompt_rel_ns = rel_stage1_ns if args.time_ref == "stage1" else rel_li9_ns

            status = "ok"
            n_before = n_after = n_neighbors = -1

            if prompt_rel_ns < prompt_min_ns:
                status = "too_early"
            elif prompt_rel_ns > max_prompt_rel_ns:
                status = "too_late"
            elif has_quality and not (
                bool(prompt_row["fit_success"])
                and np.isfinite(prompt_row["chi2_ndof"])
                and prompt_row["chi2_ndof"] < args.prompt_chi2_max
            ):
                status = "bad_prompt_fit"
            else:
                t0 = prompt_abs_ns
                t1 = t0 + args.prompt_window_ns

                # Keep hit counts around the prompt as diagnostics only.
                # They do not determine whether the prompt is isolated.
                n_before = int(
                    np.searchsorted(spill_times, t0, "left")
                    - np.searchsorted(
                        spill_times, t0 - iso_margin_ns, "left"
                    )
                )
                n_after = int(
                    np.searchsorted(
                        spill_times, t1 + iso_margin_ns, "left"
                    )
                    - np.searchsorted(spill_times, t1, "left")
                )
                iso_before_list.append(n_before)
                iso_after_list.append(n_after)

                n_neighbors = isolation_neighbor_counts.get(prompt_index, 0)
                if prompt_index in isolation_bad_indices:
                    status = "not_isolated_candidates"

            status_counter[status] += 1
            status_rows.append({
                "run": run, "spill_id": spill_id, "prompt_index": prompt_index,
                "prompt_rel_ms": prompt_rel_ns / 1e6,
                "prompt_rel_stage1_ms": rel_stage1_ns / 1e6,
                "prompt_rel_li9_ms": rel_li9_ns / 1e6,
                "n_iso_before": n_before, "n_iso_after": n_after,
                "n_prompt_neighbors": n_neighbors,
                "status": status,
            })

            if status != "ok":
                continue

            # ----------------------------------------------------------
            # Delayed search window
            # ----------------------------------------------------------
            search_start_ns = prompt_abs_ns + gap_ns
            search_stop_ns = search_start_ns + search_ns

            sl = np.searchsorted(spill_times, search_start_ns, side="left")
            sr = np.searchsorted(spill_times, search_stop_ns, side="left")
            search_times = spill_times[sl:sr]
            search_mpmt = spill_mpmt[sl:sr]
            search_pmt = spill_pmt[sl:sr]

            coverage_us = float((np.max(search_times) - search_start_ns) / 1e3) if len(search_times) else 0.0

            raw = find_delayed_candidates(
                search_times, args.window_ns, args.nhits_min, args.rms_cut_ns,
                nhits_max=args.nhits_max if args.legacy_nhits_cut else None,
            )
            n_windows += len(raw)

            merged = deduplicate_candidates(raw, args.window_ns)
            if args.legacy_nhits_cut:
                candidates = merged
            else:
                candidates = [c for c in merged if c["nhits"] <= args.nhits_max]
                n_burst_rejected += len(merged) - len(candidates)
            n_clusters += len(candidates)

            prompt_x = get_prompt_vertex(prompt_row, "v_x_fine", "v_x")
            prompt_y = get_prompt_vertex(prompt_row, "v_y_fine", "v_y")
            prompt_z = get_prompt_vertex(prompt_row, "v_z_fine", "v_z")

            spatial = []

            for candidate_id, candidate in enumerate(candidates):

                dl = np.searchsorted(search_times, candidate["time"], side="left")
                dr_ = np.searchsorted(search_times, candidate["time"] + args.window_ns, side="left")

                reco = reconstruct_delayed_candidate(
                    search_times[dl:dr_], search_mpmt[dl:dr_], search_pmt[dl:dr_]
                )

                dR = np.nan
                if all(np.isfinite(v) for v in (prompt_x, prompt_y, prompt_z,
                                                reco["delayed_x"], reco["delayed_y"], reco["delayed_z"])):
                    dR = float(np.sqrt((reco["delayed_x"] - prompt_x) ** 2
                                       + (reco["delayed_y"] - prompt_y) ** 2
                                       + (reco["delayed_z"] - prompt_z) ** 2))

                # every cluster is logged, whether or not it passes the cuts
                diag_rows.append({
                    "run": run, "spill_id": spill_id, "prompt_index": prompt_index,
                    "candidate_id": candidate_id,
                    "dR_cm": dR,
                    "fit_success": reco["fit_success"],
                    "delayed_nhits": candidate["nhits"],
                    "delayed_t_rms_ns": candidate["t_rms"],
                    "vertex_chi2_ndof": reco["vertex_chi2_ndof"],
                    "n_hits_used": reco["n_hits_used"],
                    "delta_t_us": (candidate["time"] - prompt_abs_ns) / 1e3,
                    "prompt_x": prompt_x, "prompt_y": prompt_y, "prompt_z": prompt_z,
                    "delayed_x": reco["delayed_x"], "delayed_y": reco["delayed_y"],
                    "delayed_z": reco["delayed_z"],
                    "prompt_rel_ms": prompt_rel_ns / 1e6,
                    "n_iso_before": n_before, "n_iso_after": n_after,
                    "n_prompt_neighbors": n_neighbors,
                })

                if not reco["fit_success"]:
                    continue
                if not np.isfinite(reco["vertex_chi2_ndof"]) or reco["vertex_chi2_ndof"] >= args.delayed_chi2_max:
                    continue
                if not np.isfinite(dR) or dR >= args.dr_max_cm:
                    continue

                spatial.append({"candidate_id": candidate_id, "candidate": candidate, "reco": reco, "dR": dR})

            n_spatial += len(spatial)
            selected = min(spatial, key=lambda c: c["dR"]) if spatial else None
            if selected is not None:
                n_with_neutron += 1

            # ----------------------------------------------------------
            # Prompt summary (one row per SEARCHED prompt)
            # ----------------------------------------------------------
            summary = {
                "run": run, "spill_id": spill_id, "prompt_index": prompt_index,
                "prompt_time_ns": prompt_abs_ns, "prompt_time_rel_ns": prompt_rel_ns,
                "prompt_t_ms": prompt_abs_ns / 1e6, "prompt_t_rel_ms": prompt_rel_ns / 1e6,
                "prompt_rel_stage1_ms": rel_stage1_ns / 1e6,
                "prompt_rel_li9_ms": rel_li9_ns / 1e6,
                "prompt_end_ns": prompt_abs_ns + args.prompt_window_ns,
                "search_start_ns": search_start_ns, "search_stop_ns": search_stop_ns,
                "search_start_ms": search_start_ns / 1e6, "search_stop_ms": search_stop_ns / 1e6,
                "gap_us": args.gap_us, "search_us": args.search_us,
                "coverage_us": coverage_us, "n_raw_hits_in_search": len(search_times),
                "n_delayed_windows": len(raw), "n_delayed_clusters": len(candidates),
                "n_delayed_spatial": len(spatial),
                "n_delayed_candidates": int(selected is not None),
                "has_delayed_neutron": selected is not None,
                "n_iso_before": n_before, "n_iso_after": n_after,
                "n_prompt_neighbors": n_neighbors,
                "prompt_x": prompt_x, "prompt_y": prompt_y, "prompt_z": prompt_z,
            }
            for column in ("nhits", "t_window_start_ns", "t_window_start_rel_ns", "t_window_end_ns",
                           "t_mean", "t_rms", "v_x", "v_y", "v_z", "v_x_fine", "v_y_fine", "v_z_fine",
                           "fit_success", "chi2_ndof"):
                if column in prompt_row.index:
                    summary[f"prompt_{column}"] = prompt_row[column]
            prompt_out.append(summary)

            if selected is None:
                continue

            cand, reco = selected["candidate"], selected["reco"]
            row = {
                "run": run, "spill_id": spill_id, "prompt_index": prompt_index,
                "candidate_id": selected["candidate_id"],
                "prompt_time_ns": prompt_abs_ns, "prompt_time_rel_ns": prompt_rel_ns,
                "prompt_t_ms": prompt_abs_ns / 1e6, "prompt_t_rel_ms": prompt_rel_ns / 1e6,
                "prompt_end_ns": prompt_abs_ns + args.prompt_window_ns,
                "prompt_x": prompt_x, "prompt_y": prompt_y, "prompt_z": prompt_z,
                "search_start_ns": search_start_ns, "search_stop_ns": search_stop_ns,
                "gap_us": args.gap_us, "search_us": args.search_us,
                "delta_t_us": (cand["time"] - prompt_abs_ns) / 1e3,
                "delayed_time_ns": cand["time"], "delayed_t_ms": cand["time"] / 1e6,
                "delayed_nhits": cand["nhits"], "delayed_t_rms_ns": cand["t_rms"],
                "n_overlapping_windows": cand["n_overlapping_windows"],
                "delayed_x": reco["delayed_x"], "delayed_y": reco["delayed_y"],
                "delayed_z": reco["delayed_z"],
                "vertex_chi2_ndof": reco["vertex_chi2_ndof"],
                "fit_success": reco["fit_success"], "n_hits_used": reco["n_hits_used"],
                "dR_cm": selected["dR"],
                "coverage_us": coverage_us, "n_raw_hits_in_search": len(search_times),
            }
            if "fit_error" in reco:
                row["fit_error"] = reco["fit_error"]
            delayed_out.append(row)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    output_dir = os.path.join(fv_dir, "delayed")
    os.makedirs(output_dir, exist_ok=True)
    bkg_tag = "_BKG" if args.bkg else ""
    stem = f"chunk_{args.chunk_id}{bkg_tag}{args.outtag}.pkl"

    paths = {
        "candidates": os.path.join(output_dir, f"Delayed_Li9_candidates_{stem}"),
        "prompts": os.path.join(output_dir, f"Delayed_Li9_prompts_{stem}"),
        "diagnostic": os.path.join(output_dir, f"Delayed_Li9_diagnostic_{stem}"),
        "status": os.path.join(output_dir, f"Delayed_Li9_status_{stem}"),
    }
    pd.DataFrame(delayed_out).to_pickle(paths["candidates"])
    pd.DataFrame(prompt_out).to_pickle(paths["prompts"])
    pd.DataFrame(diag_rows).to_pickle(paths["diagnostic"])
    pd.DataFrame(status_rows).to_pickle(paths["status"])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("Stage-6 summary (v2)")
    print("=" * 80)
    n_total = sum(status_counter.values())
    print(f"Stage-5 prompts in chunk                 : {n_total}")
    for key in (
        "too_early",
        "too_late",
        "bad_prompt_fit",
        "not_isolated_candidates",
        "no_raw_hits",
        "ok",
    ):
        print(f"  {key:22s}                 : {status_counter.get(key, 0)}")
    print(f"Delayed windows (>= {args.nhits_min} hits)               : {n_windows}")
    print(f"Delayed clusters kept                    : {n_clusters}")
    if not args.legacy_nhits_cut:
        print(f"Clusters rejected as bursts (> {args.nhits_max} hits)  : {n_burst_rejected}")
    print(f"Clusters passing chi2 & dR               : {n_spatial}")
    print(f"Prompts with selected neutron            : {n_with_neutron}")

    if spill_spans_ms:
        print(f"\nSanity: spill span (last hit - first hit), median = {np.median(spill_spans_ms):.1f} ms "
              f"(expected ~ 800 + {args.li9_window_ms:g} if the Li9 window ends at the last hit)")
    if iso_before_list:
        allm = np.array(iso_before_list + iso_after_list)
        print(f"Isolation margins ({args.iso_margin_us:g} us): hits per margin mean = {allm.mean():.2f}, "
              f"p50/p90/p99 = {np.percentile(allm, 50):.0f}/{np.percentile(allm, 90):.0f}/{np.percentile(allm, 99):.0f}")

    for k, v in paths.items():
        print(f"{k:10s}: {v}")
    print("=" * 80)


if __name__ == "__main__":
    main()