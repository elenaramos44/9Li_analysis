from collections import defaultdict
import numpy as np
from scipy.optimize import least_squares
import functions_bonsai

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
    **kwargs
):
    times = np.asarray(times, dtype=float)
    mpmt_ids = np.asarray(mpmt_ids, dtype=int)
    pmt_ids = np.asarray(pmt_ids, dtype=int)

    x_pmt, y_pmt, z_pmt, _ = functions_bonsai.getxyz(functions_bonsai.geo, mpmt_ids, pmt_ids)

    if drop_invalid_geo:
        good = (x_pmt > -900) & (y_pmt > -900) & (z_pmt > -900) & np.isfinite(times)
        times = times[good]
        mpmt_ids = mpmt_ids[good]
        pmt_ids = pmt_ids[good]
        x_pmt, y_pmt, z_pmt = x_pmt[good], y_pmt[good], z_pmt[good]

    # --- NEW: keep earliest hit per (mpmt_id,pmt_id) ---
    if earliest_per_channel and len(times) > 0:
        # Optional: reject very late hits globally (helps a lot for delayed candidates)
        if early_window_ns is not None:
            tmin = float(np.min(times))
            wmask = times <= (tmin + float(early_window_ns))
            times = times[wmask]
            mpmt_ids = mpmt_ids[wmask]
            pmt_ids = pmt_ids[wmask]
            x_pmt, y_pmt, z_pmt = x_pmt[wmask], y_pmt[wmask], z_pmt[wmask]

        # Find earliest time index for each channel
        # Approach: sort by time, then take first occurrence of each (mpmt,pmt).
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
    # --- END NEW ---

    if len(times) < 6:
        return {"x": np.nan, "y": np.nan, "z": np.nan, "eps": np.nan,
                "success": False, "n_hits_used": int(len(times)), "result": None}

    pmt_locs = np.column_stack([x_pmt, y_pmt, z_pmt]).astype(float)
    vc = float(c_cm_per_ns) / float(n)  # cm/ns
    sigma_ts = np.full(times.shape, float(sigma_t), dtype=float)

    # shift time origin
    t0 = float(np.min(times))
    times0 = times - t0

    # better eps guess (important)
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

    x, y, z, eps0 = result.x
    eps_abs = float(eps0 + t0)

    return {"x": float(x), "y": float(y), "z": float(z), "eps": float(eps_abs),
            "success": True, "n_hits_used": int(len(times)), "result": result}