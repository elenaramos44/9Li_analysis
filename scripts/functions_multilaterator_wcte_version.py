
import os
import sys
import warnings

import numpy as np
from scipy.stats import chi2 as chi2_distribution


# ==============================================================================
# Configuration
# ==============================================================================

# Directory containing the TimeCal repository.
# Override this path by setting the TIMECAL_PATH environment variable.
TIMECAL_PATH = os.environ.get(
    "TIMECAL_PATH",
    "/scratch/elena/TimeCal/TimeCal",
)

# Refractive index used for light propagation through water.
N_REFRACTION = 1.373

# Intrinsic timing uncertainty, in ns.
#
# The total uncertainty for each hit is calculated as:
#
#   sigma_t = sqrt(T_SIG_NS**2 + TTS**2)
#
# This value is deliberately kept as configured here; it should be validated
# against the timing calibration and conventions used in the TimeCal analysis.
T_SIG_NS = 0.01

# Minimum number of valid hits required to attempt a vertex fit.
#
# This is an internal reconstruction requirement, NOT the prompt-cluster
# selection cut. The existing stages remain responsible for applying the
# analysis-specific multiplicity requirements.
MIN_HITS = 6

# Initial vertex bounds, in mm.
POSITION_MIN_MM = -4000.0
POSITION_MAX_MM = 4000.0

# Allowed emission-time interval around the initial estimate, in ns.
EPS_BOUND_NS = 100.0

# Default scipy.optimize.least_squares loss function.
DEFAULT_LOSS = "huber"


# ==============================================================================
# Import TimeCal and WCTE geometry
# ==============================================================================

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


# ==============================================================================
# Geometry and multilaterator initialization
# ==============================================================================

_MULTILATERATOR = None
_WCD = None
_TTS_TABLE = None


def _get_multilaterator():
    """
    Initialize the WCTE geometry and TC_Multilaterator on first use.

    The initialized multilaterator and geometry are cached so that they do
    not need to be reconstructed for every cluster candidate.

    Returns
    -------
    TC_Multilaterator
        Initialized TimeCal multilaterator.
    """

    global _MULTILATERATOR, _WCD, _TTS_TABLE

    if _MULTILATERATOR is not None:
        return _MULTILATERATOR

    # Load the WCTE detector geometry.
    _WCD = geometry_wcte.load_geometry()

    # Initialize the official TimeCal multilaterator.
    _MULTILATERATOR = TC_Multilaterator(
        _WCD,
        refraction_index=N_REFRACTION,
    )

    # Build a lookup table for the design TTS of each PMT.
    #
    # The keys are (mpmt_id, pmt_id), following the indexing convention
    # used by the WCTE geometry and the existing analysis.
    _TTS_TABLE = {}

    for mpmt_id, mpmt in enumerate(_WCD.mpmts):
        for pmt_id, pmt in enumerate(mpmt.pmts):
            try:
                tts = float(pmt.prop_design["tts"])
            except (KeyError, TypeError, ValueError):
                tts = np.nan

            _TTS_TABLE[(mpmt_id, pmt_id)] = tts

    return _MULTILATERATOR


# ==============================================================================
# Utility functions
# ==============================================================================

def _failed(n_hits_used=0):
    """
    Return a consistent result dictionary for a failed reconstruction.

    All expected keys are provided, preventing downstream stages from
    encountering KeyError exceptions when a fit fails.
    """

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
    """
    Retrieve the design TTS for each hit.

    Parameters
    ----------
    mpmt_ids : array-like
        mPMT identifiers.
    pmt_ids : array-like
        PMT identifiers within each mPMT.

    Returns
    -------
    numpy.ndarray
        TTS values in ns, in the same order as the input hits.

    Notes
    -----
    The lookup assumes that the identifiers correspond to the zero-based
    indices used by geometry_wcte and by the TimeCal geometry.
    """

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
    """
    Retrieve PMT positions using the WCTE geometry.

    Parameters
    ----------
    mpmt_ids : array-like
        mPMT identifiers.
    pmt_ids : array-like
        PMT identifiers within each mPMT.

    Returns
    -------
    numpy.ndarray
        PMT positions with shape (n_hits, 3), in mm.
    """

    positions = geometry_wcte.get_xyz(
        mpmt_ids,
        pmt_ids,
        units="mm",
    )

    positions = np.asarray(positions, dtype=float)

    if positions.ndim != 2 or positions.shape != (len(mpmt_ids), 3):
        raise ValueError(
            "Unexpected PMT-position array shape. "
            f"Expected ({len(mpmt_ids)}, 3), received {positions.shape}."
        )

    return positions


# ==============================================================================
# Main reconstruction function
# ==============================================================================

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
  

    # --------------------------------------------------------------------------
    # Convert and validate input arrays
    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------
    # Initialize the multilaterator and retrieve geometry information
    # --------------------------------------------------------------------------

    ml = _get_multilaterator()

    try:
        pmt_positions = _get_pmt_positions(mpmt_ids, pmt_ids)
        tts_values = _get_tts(mpmt_ids, pmt_ids)
    except (ValueError, IndexError, KeyError, TypeError) as exc:
        warnings.warn(
            f"Could not retrieve PMT geometry or TTS information: {exc}",
            RuntimeWarning,
        )
        return _failed(n_hits_used=0)

    # --------------------------------------------------------------------------
    # Remove invalid hits
    # --------------------------------------------------------------------------

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

    # Do not remove duplicate PMT channels.
    # Every valid hit remaining in these arrays is retained for the fit.

    # --------------------------------------------------------------------------
    # Optional time-window selection
    # --------------------------------------------------------------------------

    # Disabled by default. This option is provided only for compatibility
    # with possible diagnostic studies; it is not part of the standard
    # reconstruction configuration.
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

    # --------------------------------------------------------------------------
    # Calculate per-hit timing uncertainties
    # --------------------------------------------------------------------------

    # The PMT TTS contribution is added in quadrature to the intrinsic
    # timing uncertainty.
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

    # --------------------------------------------------------------------------
    # Configure TimeCal with the valid hits
    # --------------------------------------------------------------------------

    # TC_Multilaterator.multilaterate() requires beacon_data to be assigned.
    # Since the cluster hits and their uncertainties are supplied directly
    # below, a non-None sentinel is sufficient for this interface check.
    #
    # assign_data() is deliberately not called because it would rebuild the
    # hit arrays from BeaconData rather than use the cluster arrays supplied
    # to this wrapper.
    ml.beacon_data = "cluster"
    ml.calibrated = True
    ml.use_tts = True

    # Use the WCTE PMT positions, in mm.
    ml.pmt_locs = pmt_positions

    # Shift the hit times to avoid fitting a large absolute timestamp.
    # This does not change the fitted vertex or the time residuals.
    ml.times = times - np.min(times)

    # Per-hit uncertainties, in ns.
    ml.sigma_ts = sigma_ts

    # Reset partitioning explicitly: each candidate is fitted as a single
    # cluster, without source partitioning.
    ml.partitioned = False
    ml.partition_array = None
    ml.partition = None

    # --------------------------------------------------------------------------
    # Set the initial parameter estimate and fit bounds
    # --------------------------------------------------------------------------

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

        # The analysis pipeline provides vertex coordinates in cm;
        # TimeCal expects positions in mm.
        initial_vertex_mm = initial_vertex_cm * 10.0

    # Estimate the emission time from the initial spatial position.
    distances_mm = np.linalg.norm(
        pmt_positions - initial_vertex_mm,
        axis=1,
    )

    # The propagation speed used here is the same as in TC_Multilaterator.
    light_velocity_mm_ns = ml.vc

    if not np.isfinite(light_velocity_mm_ns) or light_velocity_mm_ns <= 0:
        return _failed(n_hits_used=n_hits)

    tof_initial_ns = distances_mm / light_velocity_mm_ns

    eps_guess_ns = np.median(
        ml.times - tof_initial_ns
    )

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

    # --------------------------------------------------------------------------
    # Run the TimeCal multilateration fit
    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------
    # Validate the fit result
    # --------------------------------------------------------------------------

    if result is None or not result.success:
        failed_result = _failed(n_hits_used=n_hits)

        if result is not None:
            failed_result["result"] = result

        return failed_result

    if not np.isfinite(result.x).all():
        failed_result = _failed(n_hits_used=n_hits)
        failed_result["result"] = result
        return failed_result

    # --------------------------------------------------------------------------
    # Calculate fit diagnostics
    # --------------------------------------------------------------------------

    # TC_Multilaterator.rho() returns normalized residuals:
    #
    #   pull_i = (measured_time_i - fitted_emission_time - TOF_i) / sigma_i
    #
    # Preserve this meaning for the "pulls" output field.
    pulls = np.asarray(ml.rho(result.x), dtype=float)

    if pulls.shape != (n_hits,) or not np.isfinite(pulls).all():
        failed_result = _failed(n_hits_used=n_hits)
        failed_result["result"] = result
        return failed_result

    # Convert normalized residuals to time residuals in ns.
    residuals_ns = pulls * sigma_ts

    # Calculate chi-square independently of the covariance inversion in
    # TC_Multilaterator.multilaterate(). This avoids depending on whether
    # its internal Hessian inversion succeeds.
    chi2_value = float(np.sum(pulls**2))

    # The fit has four parameters: x, y, z, and the emission-time offset.
    ndof = int(n_hits - 4)

    if ndof > 0:
        chi2_ndof = chi2_value / ndof
        p_value = float(chi2_distribution.sf(chi2_value, ndof))
    else:
        chi2_ndof = np.nan
        p_value = np.nan

    # --------------------------------------------------------------------------
    # Convert the reconstructed position from mm to cm
    # --------------------------------------------------------------------------

    x_cm = float(result.x[0] / 10.0)
    y_cm = float(result.x[1] / 10.0)
    z_cm = float(result.x[2] / 10.0)

    eps_ns = float(result.x[3])

    # --------------------------------------------------------------------------
    # Return results
    # --------------------------------------------------------------------------

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


# ==============================================================================
# Backward-compatible alias
# ==============================================================================

def run_multilateration_official(*args, **kwargs):
    """
    Backward-compatible alias for run_multilateration_candidate().
    """

    return run_multilateration_candidate(*args, **kwargs)