#!/usr/bin/env python3
"""
geometry_wcte.py

Utilities for loading the official WCTE detector geometry from the
Geometry_WCTE package.

This module provides a single source of truth for PMT positions.
Coordinates are returned in the official WCTE coordinate convention
(beam pipe at y = 0 cm).

The geometry is loaded only once and cached in memory.
"""

import sys
import numpy as np

# Path to the official WCTE geometry package
sys.path.insert(0, "/scratch/elena/Geometry_WCTE")

from Geometry.Device import Device


#-------------------------------------------------------------------------
# Global geometry cache


_HALL = None
_WCD = None

# Dictionary:
#
#     (mpmt_id, pmt_id)
#            ↓
#      (x, y, z)   [mm]
#
_PMT_POS = {}


#-------------------------------------------------------------------------
# Geometry loader


def load_geometry(
    geo_file="/scratch/elena/Geometry_WCTE/examples/wcte_bldg157.geo",
    wcd_index=0,
):
    """
    Load the official WCTE geometry.

    The geometry is loaded only once.
    Subsequent calls simply return the cached detector.

    Parameters
    ----------
    geo_file : str
        Path to the official .geo file.

    wcd_index : int
        Which WCD inside the geometry file to use.

    Returns
    -------
    WCD object
    """

    global _HALL, _WCD, _PMT_POS

    if _WCD is not None:
        return _WCD

    _HALL = Device.open_file(geo_file)
    _WCD = _HALL.wcds[wcd_index]

    _PMT_POS.clear()

    for mpmt_id, mpmt in enumerate(_WCD.mpmts):

        for pmt_id, pmt in enumerate(mpmt.pmts):

            loc = pmt.get_placement("design", _WCD)["location"]

            _PMT_POS[(mpmt_id, pmt_id)] = (
                float(loc[0]),
                float(loc[1]),
                float(loc[2]),
            )

    return _WCD


#-------------------------------------------------------------------------
# Coordinate access


def get_xyz(mpmt_ids, pmt_ids, units="mm"):
    """
    Retrieve PMT coordinates.

    Parameters
    ----------
    mpmt_ids : array-like
    pmt_ids : array-like

    units : {"mm","cm"}

    Returns
    -------
    x, y, z : numpy.ndarray
    """

    load_geometry()

    mpmt_ids = np.asarray(mpmt_ids, dtype=int)
    pmt_ids = np.asarray(pmt_ids, dtype=int)

    x = np.full(len(mpmt_ids), np.nan)
    y = np.full(len(mpmt_ids), np.nan)
    z = np.full(len(mpmt_ids), np.nan)

    for i, (m, p) in enumerate(zip(mpmt_ids, pmt_ids)):

        pos = _PMT_POS.get((m, p))

        if pos is None:
            continue

        x[i], y[i], z[i] = pos

    if units == "cm":

        x /= 10.0
        y /= 10.0
        z /= 10.0

    elif units != "mm":

        raise ValueError("units must be either 'mm' or 'cm'")

    return x, y, z

#-------------------------------------------------------------------------

def get_all_pmts(units="cm"):

    load_geometry()

    x = []
    y = []
    z = []

    for pos in _PMT_POS.values():
        x.append(pos[0])
        y.append(pos[1])
        z.append(pos[2])

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    if units == "cm":
        x /= 10
        y /= 10
        z /= 10

    return x, y, z