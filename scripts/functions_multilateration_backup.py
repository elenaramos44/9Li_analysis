#!/usr/bin/env python3
from collections import defaultdict
import numpy as np
from scipy.optimize import least_squares
import functions_bonsai
import sys

# Append the custom official geometry tools path tailored exactly to your workspace
sys.path.insert(0, "/scratch/elena/Geometry_WCTE")
from Geometry.Device import Device

# --- GLOBAL GEOMETRY CACHE AND BONSAIDIR LOOKUP ASSIGNMENTS ---
geo = None          # Internal global table mapping for legacy functions
_HALL = None
_WCD = None
_PMT_POS = {}       # Fast tracking dict: (mpmt_id, pmt_id) -> (x, y, z) [in mm]

def _load_wcte_geo(
    geo_file="/scratch/elena/Geometry_WCTE/examples/wcte_bldg157.geo",
    wcd_index=0,
):
    """
    Load the official WCTE geometry (.geo) and cache all PMT coordinates.

    Coordinates are returned directly in the official WCTE convention
    (beam pipe at y = 0). No manual coordinate shifts are applied.
    """

    global _HALL, _WCD, _PMT_POS

    if _WCD is not None:
        return _WCD

    _HALL = Device.open_file(geo_file)
    _WCD = _HALL.wcds[wcd_index]

    _PMT_POS = {}

    for mpmt_id, mpmt in enumerate(_WCD.mpmts):
        for pmt_id, pmt in enumerate(mpmt.pmts):

            loc = pmt.get_placement("design")["location"]

            _PMT_POS[(mpmt_id, pmt_id)] = (
                float(loc[0]),
                float(loc[1]),
                float(loc[2]),
            )

    return _WCD

def _get_xyz_from_wcte(mpmt_ids, pmt_ids):
    """Retrieves high-precision coordinates directly from the cached .geo structure."""
    x = np.full(len(mpmt_ids), np.nan, dtype=float)
    y = np.full(len(mpmt_ids), np.nan, dtype=float)
    z = np.full(len(mpmt_ids), np.nan, dtype=float)

    for i, (m, p) in enumerate(zip(mpmt_ids, pmt_ids)):
        tup = _PMT_POS.get((int(m), int(p)))
        if tup is None:
            continue
        x[i], y[i], z[i] = tup
    return x, y, z


# =====================================================================
# METHOD 1: Time Calibration Multilateration Engine (Partner's Version)
# =====================================================================
def run_multilateration_timecal(
    times, mpmt_ids, pmt_ids,
    *,
    sigma_t=1.0,
    n=1.33,
    c_mm_per_ns=299.792458,   # mm/ns
    guess=(0., 0., 0., 0.),
    mins=(-3000., -3000., -3000., -3000.),
    maxs=(3000., 3000., 3000., 3000.),
    drop_invalid_geo=True,
    earliest_per_channel=True,
    early_window_ns=100.0,
    robust_loss="soft_l1",
    f_scale=2.0,
    geo_file="/scratch/elena/Geometry_WCTE/examples/wcte_bldg157.geo",
    placement="design",
    **kwargs
):
    times = np.asarray(times, dtype=float)
    mpmt_ids = np.asarray(mpmt_ids, dtype=int)
    pmt_ids = np.asarray(pmt_ids, dtype=int)

    # ---- NEW: geometry comes from .geo ----
    _load_wcte_geo(geo_file=geo_file, wcd_index=0)
    x_pmt, y_pmt, z_pmt = _get_xyz_from_wcte(mpmt_ids, pmt_ids)

    if drop_invalid_geo:
        good = np.isfinite(x_pmt) & np.isfinite(y_pmt) & np.isfinite(z_pmt) & np.isfinite(times)
        times = times[good]
        mpmt_ids = mpmt_ids[good]
        pmt_ids = pmt_ids[good]
        x_pmt, y_pmt, z_pmt = x_pmt[good], y_pmt[good], z_pmt[good]

    # keep your old hit filtering exactly
    if earliest_per_channel and len(times) > 0:
        if early_window_ns is not None:
            tmin = float(np.min(times))
            wmask = times <= (tmin + float(early_window_ns))
            times = times[wmask]
            mpmt_ids = mpmt_ids[wmask]
            pmt_ids = pmt_ids[wmask]
            x_pmt, y_pmt, z_pmt = x_pmt[wmask], y_pmt[wmask], z_pmt[wmask]

        order = np.argsort(times)
        times_s = times[order]
        mpmt_s = mpmt_ids[order]
        pmt_s = pmt_ids[order]
        x_s, y_s, z_s = x_pmt[order], y_pmt[order], z_pmt[order]

        seen = set()
        keep_idx = []
        for i, key in enumerate(zip(mpmt_s, pmt_s)):
            if key in seen:
                continue
            seen.add(key)
            keep_idx.append(i)

        keep_idx = np.asarray(keep_idx, dtype=int)
        times = times_s[keep_idx]
        mpmt_ids = mpmt_s[keep_idx]
        pmt_ids = pmt_s[keep_idx]
        x_pmt, y_pmt, z_pmt = x_s[keep_idx], y_s[keep_idx], z_s[keep_idx]

    if len(times) < 6:
        return {"x": np.nan, "y": np.nan, "z": np.nan, "eps": np.nan,
                "success": False, "n_hits_used": int(len(times)), "result": None}

    pmt_locs = np.column_stack([x_pmt, y_pmt, z_pmt]).astype(float)  # mm
    vc = float(c_mm_per_ns) / float(n)                               # mm/ns
    sigma_ts = np.full(times.shape, float(sigma_t), dtype=float)

    t0 = float(np.min(times))
    times0 = times - t0

    loc0 = np.array([0., 0., 0.])
    tof0 = np.linalg.norm(pmt_locs - loc0, axis=1) / vc
    eps_guess = float(np.median(times0 - tof0))
    x0 = np.array([0., 0., 0., eps_guess], dtype=float)

    def rho(pars):
        loc = pars[0:3]
        eps = pars[3]
        dists = np.linalg.norm(pmt_locs - loc, axis=1)
        tofs = dists / vc
        return (times0 - eps - tofs) / sigma_ts

    def jac(pars):
        loc = pars[0:3]
        light_vecs = pmt_locs - loc
        dists = np.linalg.norm(light_vecs, axis=1)
        dists = np.where(dists == 0, 1e-12, dists)
        jac_xyz = light_vecs / dists.reshape(-1, 1) / vc / sigma_ts.reshape(-1, 1)
        jac_eps = -1.0 / sigma_ts
        return np.column_stack([jac_xyz, jac_eps])

    loss = robust_loss if robust_loss is not None else "linear"

    result = least_squares(
        rho, x0, jac,
        bounds=(np.array(mins, dtype=float), np.array(maxs, dtype=float)),
        loss=loss,
        f_scale=float(f_scale),
        **kwargs
    )

    if not result.success:
        return {"x": np.nan, "y": np.nan, "z": np.nan, "eps": np.nan,
                "success": False, "n_hits_used": int(len(times)), "result": result}

    pulls = rho(result.x)
    chi2 = float(np.sum(pulls**2))
    ndof = int(len(pulls) - 4)
    chi2_ndof = float(chi2 / ndof) if ndof > 0 else np.inf

    x, y, z, eps0 = result.x
    eps_abs = float(eps0 + t0)

    return {
        "x": float(x), "y": float(y), "z": float(z), "eps": float(eps_abs),
        "success": True, "n_hits_used": int(len(pulls)),
        "chi2": chi2, "ndof": ndof, "chi2_ndof": chi2_ndof, "result": result,
    }


# =====================================================================
# METHOD 2: Fine Candidate Clustering Engine (Your Standard Reference)
# =====================================================================
def run_multilateration_candidate(
    times, mpmt_ids, pmt_ids,
    *,
    sigma_t=1.0,
    n=1.33,
    c_cm_per_ns=29.9792458,   # cm/ns
    guess=(0., 0., 0., 0.),
    mins=(-300., -300., -300., -300.),
    maxs=(300., 300., 300., 300.),
    drop_invalid_geo=True,
    earliest_per_channel=True,
    early_window_ns=100.0,    # set None to disable extra windowing
    robust_loss="soft_l1",
    f_scale=2.0,
    geo_file="/scratch/elena/Geometry_WCTE/examples/wcte_bldg157.geo",
    **kwargs
):
    times = np.asarray(times, dtype=float)
    mpmt_ids = np.asarray(mpmt_ids, dtype=int)
    pmt_ids = np.asarray(pmt_ids, dtype=int)

    # Ensure geometry table maps are built/loaded locally using your paths before calling getxyz
    _load_wcte_geo(geo_file=geo_file)

    x_pmt, y_pmt, z_pmt = _get_xyz_from_wcte(
        mpmt_ids,
        pmt_ids,
    )

    if drop_invalid_geo:
        good = (
            np.isfinite(x_pmt)
            & np.isfinite(y_pmt)
            & np.isfinite(z_pmt)
            & np.isfinite(times)
        )
        times = times[good]
        mpmt_ids = mpmt_ids[good]
        pmt_ids = pmt_ids[good]
        x_pmt, y_pmt, z_pmt = x_pmt[good], y_pmt[good], z_pmt[good]

    if earliest_per_channel and len(times) > 0:
        if early_window_ns is not None:
            tmin = float(np.min(times))
            wmask = times <= (tmin + float(early_window_ns))
            times = times[wmask]
            mpmt_ids = mpmt_ids[wmask]
            pmt_ids = pmt_ids[wmask]
            x_pmt, y_pmt, z_pmt = x_pmt[wmask], y_pmt[wmask], z_pmt[wmask]

        order = np.argsort(times)
        times_s = times[order]
        mpmt_s = mpmt_ids[order]
        pmt_s = pmt_ids[order]
        x_s, y_s, z_s = x_pmt[order], y_pmt[order], z_pmt[order]

        seen = set()
        keep_idx = []
        for i, key in enumerate(zip(mpmt_s, pmt_s)):
            if key in seen:
                continue
            seen.add(key)
            keep_idx.append(i)

        keep_idx = np.asarray(keep_idx, dtype=int)
        times = times_s[keep_idx]
        mpmt_ids = mpmt_s[keep_idx]
        pmt_ids = pmt_s[keep_idx]
        x_pmt, y_pmt, z_pmt = x_s[keep_idx], y_s[keep_idx], z_s[keep_idx]

    if len(times) < 6:
        return {"x": np.nan, "y": np.nan, "z": np.nan, "eps": np.nan,
                "success": False, "n_hits_used": int(len(times)), "result": None}


    # Convert official WCTE geometry from mm to cm
    x_pmt /= 10.0
    y_pmt /= 10.0
    z_pmt /= 10.0

    pmt_locs = np.column_stack([x_pmt, y_pmt, z_pmt]).astype(float)
    vc = float(c_cm_per_ns) / float(n)  # cm/ns
    sigma_ts = np.full(times.shape, float(sigma_t), dtype=float)

    t0 = float(np.min(times))
    times0 = times - t0

    loc0 = np.array([0., 0., 0.])
    tof0 = np.linalg.norm(pmt_locs - loc0, axis=1) / vc
    eps_guess = float(np.median(times0 - tof0))
    x0 = np.array([0., 0., 0., eps_guess], dtype=float)

    def rho(pars):
        loc = pars[0:3]
        eps = pars[3]
        dists = np.linalg.norm(pmt_locs - loc, axis=1)
        tofs = dists / vc
        return (times0 - eps - tofs) / sigma_ts

    def jac(pars):
        loc = pars[0:3]
        light_vecs = pmt_locs - loc
        dists = np.linalg.norm(light_vecs, axis=1)
        dists = np.where(dists == 0, 1e-12, dists)
        jac_xyz = light_vecs / dists.reshape(-1, 1) / vc / sigma_ts.reshape(-1, 1)
        jac_eps = -1.0 / sigma_ts
        return np.column_stack([jac_xyz, jac_eps])

    loss = robust_loss if robust_loss is not None else "linear"

    result = least_squares(
        rho, x0, jac,
        bounds=(np.array(mins, dtype=float), np.array(maxs, dtype=float)),
        loss=loss,
        f_scale=float(f_scale),
        **kwargs
    )

    if not result.success:
        return {"x": np.nan, "y": np.nan, "z": np.nan, "eps": np.nan,
                "success": False, "n_hits_used": int(len(times)), "result": result}

    pulls = rho(result.x)
    chi2 = float(np.sum(pulls**2))
    ndof = int(len(pulls) - 4)
    chi2_ndof = float(chi2 / ndof) if ndof > 0 else np.inf

    x, y, z, eps0 = result.x
    eps_abs = float(eps0 + t0)

    return {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "eps": float(eps_abs),
        "success": True,
        "n_hits_used": int(len(pulls)),
        "chi2": chi2,
        "ndof": ndof,
        "chi2_ndof": chi2_ndof,
        "pulls": pulls,
        "result": result,
    }


# =====================================================================
# METHOD 3: Coarse/Fine Grid Space Scanning Vertex Finder
# =====================================================================
def run_grid_vertex_candidate(
    times, mpmt_ids, pmt_ids,
    *,
    n=1.33,
    c_cm_per_ns=29.9792458,      # cm/ns
    xyz_bounds_cm=300.0,         # global search volume
    coarse_step_cm=10.0,
    fine_step_cm=1.0,
    refine_halfwidth_cm=20.0,    # +/- around best coarse
    dt_cut_ns=3.0,               # cut used in fine stage (set None to disable)
    earliest_per_channel=True,   # recommended
    drop_invalid_geo=True,
    geo_file="/scratch/elena/Geometry_WCTE/examples/wcte_bldg157.geo",
):
    times = np.asarray(times, dtype=float)
    mpmt_ids = np.asarray(mpmt_ids, dtype=int)
    pmt_ids = np.asarray(pmt_ids, dtype=int)

    _load_wcte_geo(geo_file=geo_file)

    x_pmt, y_pmt, z_pmt = _get_xyz_from_wcte(
        mpmt_ids,
        pmt_ids,
    )

    if drop_invalid_geo:
        good = (
            np.isfinite(x_pmt)
            & np.isfinite(y_pmt)
            & np.isfinite(z_pmt)
            & np.isfinite(times)
        )
        times = times[good]
        mpmt_ids = mpmt_ids[good]
        pmt_ids = pmt_ids[good]
        x_pmt, y_pmt, z_pmt = x_pmt[good], y_pmt[good], z_pmt[good]

    if len(times) < 6:
        return {"x": np.nan, "y": np.nan, "z": np.nan, "t0": np.nan,
                "trms": np.nan, "chi2": np.nan, "ndof": -1,
                "success": False, "n_hits_used": int(len(times))}

    if earliest_per_channel:
        order = np.argsort(times)
        times_s = times[order]
        mpmt_s = mpmt_ids[order]
        pmt_s = pmt_ids[order]
        x_s, y_s, z_s = x_pmt[order], y_pmt[order], z_pmt[order]

        seen = set()
        keep = []
        for i, key in enumerate(zip(mpmt_s, pmt_s)):
            if key in seen:
                continue
            seen.add(key)
            keep.append(i)
        keep = np.asarray(keep, dtype=int)

        times = times_s[keep]
        x_pmt, y_pmt, z_pmt = x_s[keep], y_s[keep], z_s[keep]

    tmin = float(np.min(times))
    times0 = times - tmin

    x_pmt /= 10.0
    y_pmt /= 10.0
    z_pmt /= 10.0

    pmt_locs = np.column_stack([x_pmt, y_pmt, z_pmt]).astype(float)
    vc = float(c_cm_per_ns) / float(n)  # cm/ns

    def score_vertex(vxyz, use_cut=False):
        vxyz = np.asarray(vxyz, dtype=float)
        dists = np.linalg.norm(pmt_locs - vxyz, axis=1)
        tofs = dists / vc
        t_corr = times0 - tofs

        t0 = float(np.median(t_corr))
        dt = t_corr - t0

        if use_cut and (dt_cut_ns is not None):
            mask = np.abs(dt) < float(dt_cut_ns)
            if np.count_nonzero(mask) < 6:
                return (np.inf, t0, np.inf, -1, int(np.count_nonzero(mask)))
            dt_use = dt[mask]
        else:
            dt_use = dt

        trms = float(np.sqrt(np.mean(dt_use**2)))
        chi2 = float(np.sum(dt_use**2))
        ndof = int(len(dt_use) - 4)
        return (trms, t0, chi2, ndof, int(len(dt_use)))

    def make_grid(xmin, xmax, step):
        return np.arange(xmin, xmax + 0.5*step, step)

    # --- Stage 1: coarse global grid ---
    xs = make_grid(-xyz_bounds_cm, xyz_bounds_cm, coarse_step_cm)
    best = (np.inf, None, None, None, None)
    for x in xs:
        for y in xs:
            for z in xs:
                trms, t0, chi2, ndof, n_used = score_vertex((x, y, z), use_cut=False)
                if trms < best[0]:
                    best = (trms, (float(x), float(y), float(z)), t0, chi2, (ndof, n_used))

    coarse_trms, (bx, by, bz), coarse_t0, coarse_chi2, (coarse_ndof, coarse_nused) = best

    # --- Stage 2: fine grid around coarse best ---
    fxs = make_grid(bx - refine_halfwidth_cm, bx + refine_halfwidth_cm, fine_step_cm)
    fys = make_grid(by - refine_halfwidth_cm, by + refine_halfwidth_cm, fine_step_cm)
    fzs = make_grid(bz - refine_halfwidth_cm, bz + refine_halfwidth_cm, fine_step_cm)
    
    best2 = (np.inf, None, None, None, None)
    for x in fxs:
        for y in fys:
            for z in fzs:
                trms, t0, chi2, ndof, n_used = score_vertex((x, y, z), use_cut=True)
                if trms < best2[0]:
                    best2 = (trms, (float(x), float(y), float(z)), t0, chi2, (ndof, n_used))

    if best2[1] is None:
        best2 = (np.inf, None, None, None, None)
        for x in fxs:
            for y in fys:
                for z in fzs:
                    trms, t0, chi2, ndof, n_used = score_vertex((x, y, z), use_cut=False)
                    if trms < best2[0]:
                        best2 = (trms, (float(x), float(y), float(z)), t0, chi2, (ndof, n_used))

    fine_trms, (fx, fy, fz), fine_t0, fine_chi2, (fine_ndof, fine_nused) = best2
    t0_abs = float(fine_t0 + tmin)

    return {
        "x": float(fx), "y": float(fy), "z": float(fz), "t0": t0_abs,
        "trms": float(fine_trms), "chi2": float(fine_chi2), "ndof": int(fine_ndof),
        "chi2_ndof": float(fine_chi2 / fine_ndof) if fine_ndof > 0 else np.inf,
        "success": np.isfinite(fine_trms), "n_hits_used": int(fine_nused),
        "coarse": {
            "x": float(bx), "y": float(by), "z": float(bz), "trms": float(coarse_trms),
            "t0": float(coarse_t0 + tmin), "chi2": float(coarse_chi2),
            "ndof": int(coarse_ndof), "n_hits_used": int(coarse_nused),
        }
    }