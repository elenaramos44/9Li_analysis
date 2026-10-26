import os
import sys
import warnings

import numpy as np
from scipy.stats import chi2 as chi2_distribution

TIMECAL_PATH = os.environ.get("TIMECAL_PATH", "/scratch/elena/TimeCal/TimeCal")

N_REFRACTION = 1.373
T_SIG_NS = 0.01
MIN_HITS = 6
POSITION_MIN_MM = -4000.0
POSITION_MAX_MM = 4000.0
EPS_BOUND_NS = 100.0
DEFAULT_LOSS = "huber"

if TIMECAL_PATH not in sys.path:
    sys.path.insert(0, TIMECAL_PATH)

try:
    from TC_Multilaterator import TC_Multilaterator
except ImportError as exc:
    raise ImportError(
        "Could not import TC_Multilaterator from the TimeCal repository. "
        f"Expected to find TC_Multilaterator.py under: {TIMECAL_PATH}. "
        "Check TIMECAL_PATH and the repository contents."
    ) from exc

try:
    import geometry_wcte
except ImportError as exc:
    raise ImportError(
        "Could not import geometry_wcte. Make sure the WCTE analysis "
        "scripts directory is available in PYTHONPATH."
    ) from exc


_MULTILATERATOR = None
_WCD = None
_TTS_TABLE = None


def _get_multilaterator():
    global _MULTILATERATOR, _WCD, _TTS_TABLE

    if _MULTILATERATOR is not None:
        return _MULTILATERATOR

    _WCD = geometry_wcte.load_geometry()

    _MULTILATERATOR = TC_Multilaterator(
        _WCD,
        place_info="est",
        refraction_index=N_REFRACTION,
    )

    _TTS_TABLE = {}

    for mpmt_id, mpmt in enumerate(_WCD.mpmts):
        for pmt_id, pmt in enumerate(mpmt.pmts):
            try:
                tts = float(pmt.prop_design["tts"])
            except (KeyError, TypeError, ValueError):
                tts = np.nan

            _TTS_TABLE[(mpmt_id, pmt_id)] = tts

    return _MULTILATERATOR


def _failed(n_hits_used=0):
    return {
        "x": np.nan,
        "y": np.nan,
        "z": np.nan,
        "eps": np.nan,
        "success": False,
        "n_hits_used": int(n_hits_used),
        "chi2": np.nan,
        "ndof": 0,
        "chi2_ndof": np.nan,
        "p_value": np.nan,
        "pulls": np.array([], dtype=float),
        "residuals_ns": np.array([], dtype=float),
        "result": None,
    }


def _get_tts(mpmt_ids, pmt_ids):
    _get_multilaterator()

    tts_values = np.full(len(mpmt_ids), np.nan, dtype=float)

    for i, (mpmt_id, pmt_id) in enumerate(zip(mpmt_ids, pmt_ids)):
        try:
            key = (int(mpmt_id), int(pmt_id))
            tts_values[i] = _TTS_TABLE.get(key, np.nan)
        except (TypeError, ValueError):
            continue

    return tts_values


def _get_pmt_positions(mpmt_ids, pmt_ids):
    x, y, z = geometry_wcte.get_xyz(mpmt_ids, pmt_ids, units="mm")
    positions = np.column_stack([x, y, z]).astype(float)

    if positions.shape != (len(mpmt_ids), 3):
        raise ValueError(
            "Unexpected PMT-position array shape. "
            f"Expected ({len(mpmt_ids)}, 3), received {positions.shape}."
        )

    return positions


def run_multilateration_candidate(
    times,
    mpmt_ids,
    pmt_ids,
    sigma_t=1.0,
    early_window_ns=100.0,
    robust_loss=DEFAULT_LOSS,
    initial_vertex=None,
    use_peak_window=False,
    **least_squares_kwargs,
):
    times = np.asarray(times, dtype=float)
    mpmt_ids = np.asarray(mpmt_ids)
    pmt_ids = np.asarray(pmt_ids)

    if not (len(times) == len(mpmt_ids) == len(pmt_ids)):
        raise ValueError(
            "times, mpmt_ids, and pmt_ids must have the same length."
        )

    if times.ndim != 1 or mpmt_ids.ndim != 1 or pmt_ids.ndim != 1:
        raise ValueError(
            "times, mpmt_ids, and pmt_ids must be one-dimensional arrays."
        )

    if len(times) == 0:
        return _failed(n_hits_used=0)

    ml = _get_multilaterator()

    try:
        pmt_positions = _get_pmt_positions(mpmt_ids, pmt_ids)
        tts_values = _get_tts(mpmt_ids, pmt_ids)
    except (ValueError, IndexError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Could not retrieve PMT geometry or TTS information: {exc}"
        ) from exc

    valid_mask = (
        np.isfinite(times)
        & np.isfinite(pmt_positions).all(axis=1)
        & np.isfinite(tts_values)
        & (tts_values >= 0.0)
    )

    times = times[valid_mask]
    mpmt_ids = mpmt_ids[valid_mask]
    pmt_ids = pmt_ids[valid_mask]
    pmt_positions = pmt_positions[valid_mask]
    tts_values = tts_values[valid_mask]

    if use_peak_window and len(times) > 0:
        t_min = np.min(times)

        peak_mask = (
            (times >= t_min)
            & (times <= t_min + float(early_window_ns))
        )

        times = times[peak_mask]
        mpmt_ids = mpmt_ids[peak_mask]
        pmt_ids = pmt_ids[peak_mask]
        pmt_positions = pmt_positions[peak_mask]
        tts_values = tts_values[peak_mask]

    n_hits = len(times)

    if n_hits < MIN_HITS:
        return _failed(n_hits_used=n_hits)

    sigma_ts = np.sqrt(T_SIG_NS**2 + tts_values**2)

    valid_sigma = np.isfinite(sigma_ts) & (sigma_ts > 0.0)

    if not np.all(valid_sigma):
        times = times[valid_sigma]
        mpmt_ids = mpmt_ids[valid_sigma]
        pmt_ids = pmt_ids[valid_sigma]
        pmt_positions = pmt_positions[valid_sigma]
        tts_values = tts_values[valid_sigma]
        sigma_ts = sigma_ts[valid_sigma]

    n_hits = len(times)

    if n_hits < MIN_HITS:
        return _failed(n_hits_used=n_hits)

    ml.beacon_data = "cluster"
    ml.calibrated = True
    ml.use_tts = True
    ml.pmt_locs = pmt_positions
    ml.times = times - np.min(times)
    ml.sigma_ts = sigma_ts
    ml.partitioned = False
    ml.partition_array = None
    ml.partition = None

    if initial_vertex is None:
        initial_vertex_mm = np.array([0.0, 0.0, 0.0], dtype=float)
    else:
        initial_vertex_cm = np.asarray(initial_vertex, dtype=float)

        if initial_vertex_cm.shape != (3,):
            raise ValueError(
                "initial_vertex must contain exactly three coordinates "
                "in cm: (x, y, z)."
            )

        if not np.isfinite(initial_vertex_cm).all():
            raise ValueError("initial_vertex must contain finite coordinates.")

        initial_vertex_mm = initial_vertex_cm * 10.0

    distances_mm = np.linalg.norm(
        pmt_positions - initial_vertex_mm,
        axis=1,
    )

    light_velocity_mm_ns = ml.vc

    if not np.isfinite(light_velocity_mm_ns) or light_velocity_mm_ns <= 0:
        return _failed(n_hits_used=n_hits)

    tof_initial_ns = distances_mm / light_velocity_mm_ns

    eps_guess_ns = np.median(ml.times - tof_initial_ns)

    guess = (
        float(initial_vertex_mm[0]),
        float(initial_vertex_mm[1]),
        float(initial_vertex_mm[2]),
        float(eps_guess_ns),
    )

    mins_fit = (
        POSITION_MIN_MM,
        POSITION_MIN_MM,
        POSITION_MIN_MM,
        eps_guess_ns - EPS_BOUND_NS,
    )

    maxs_fit = (
        POSITION_MAX_MM,
        POSITION_MAX_MM,
        POSITION_MAX_MM,
        eps_guess_ns + EPS_BOUND_NS,
    )

    try:
        result, _, _, _ = ml.multilaterate(
            guess=guess,
            mins=mins_fit,
            maxs=maxs_fit,
            method="trf",
            loss=robust_loss,
            **least_squares_kwargs,
        )
    except Exception as exc:
        warnings.warn(
            f"TimeCal multilateration raised an exception: {exc}",
            RuntimeWarning,
        )
        return _failed(n_hits_used=n_hits)

    if result is None or not result.success:
        failed_result = _failed(n_hits_used=n_hits)

        if result is not None:
            failed_result["result"] = result

        return failed_result

    if not np.isfinite(result.x).all():
        failed_result = _failed(n_hits_used=n_hits)
        failed_result["result"] = result
        return failed_result

    pulls = np.asarray(ml.rho(result.x), dtype=float)

    if pulls.shape != (n_hits,) or not np.isfinite(pulls).all():
        failed_result = _failed(n_hits_used=n_hits)
        failed_result["result"] = result
        return failed_result

    residuals_ns = pulls * sigma_ts

    chi2_value = float(np.sum(pulls**2))

    ndof = int(n_hits - 4)

    if ndof > 0:
        chi2_ndof = chi2_value / ndof
        p_value = float(chi2_distribution.sf(chi2_value, ndof))
    else:
        chi2_ndof = np.nan
        p_value = np.nan

    x_cm = float(result.x[0] / 10.0)
    y_cm = float(result.x[1] / 10.0)
    z_cm = float(result.x[2] / 10.0)

    eps_ns = float(result.x[3])

    return {
        "x": x_cm,
        "y": y_cm,
        "z": z_cm,
        "eps": eps_ns,
        "success": bool(result.success),
        "n_hits_used": int(n_hits),
        "chi2": chi2_value,
        "ndof": ndof,
        "chi2_ndof": chi2_ndof,
        "p_value": p_value,
        "pulls": pulls,
        "residuals_ns": residuals_ns,
        "result": result,
    }


def run_multilateration_official(*args, **kwargs):
    return run_multilateration_candidate(*args, **kwargs)