# This file is part of SMARTIES.
# Copyright (C) 2024-2026 Members of the Simons Observatory collaboration.
# Please refer to the LICENSE file in the root of this repository.

import logging
import numpy as np
import healpy as hp
from pixell import enmap

from smarties.utils.harmonics import map2alm_anypix
from smarties.external.s4cmb import get_second_spin_derivative, get_second_spherical_derivatives_from_spin_derivatives

logger = logging.getLogger(__name__)

__all__ = [
    'get_all_spherical_derivatives_from_temperature_maps',
]

def get_all_spherical_derivatives_from_temperature_maps(
    input_map: np.ndarray | enmap.ndmap,
    lmax: int,
    input_alm_to_use: np.ndarray = None,
    n_iter: int = 10,
):
    """
    Helper function to obain spherical derivatives from temperature maps. 

    Parameters
    ----------
    input_map : np.ndarray | enmap.ndmap
        Input map in either CAR (enmap.ndmap) or HEALPIX (np.ndarray) format. 
        The first field is assumed to be the intensity field. 
    lmax : int
        Maximum multipole moment.
    input_alm_to_use : np.ndarray, optional
        Pre-computed alms to use for the spherical derivatives. If not provided, they will be computed from the input map.
    niter : int, optional
        Number of iterations for the map2alm function (default is 10, only relevant for healpix).

    Returns
    -------
    spherical_derivatives_dict : dict
        Dictionary containing keys for both first and second derivatives, with keys being:
        * 'phi': first derivative with respect to phi (including factor 1/sin(theta)) 
        * 'theta': first derivative with respect to theta
        * 'theta_theta': second derivative with respect to theta
        * 'phi_phi': second derivative with respect to phi (including factor 1/sin^2(theta))
        * 'theta_phi': mixed second derivative with respect to theta and phi (including factor 1/sin(theta))

    """

    intensity_full_map = input_map[0] if input_map.ndim > 1 else input_map
    input_spin = 0

    shape_pixels_output = intensity_full_map.shape

    if input_alm_to_use is None:
        alms_I = map2alm_anypix(
            intensity_full_map, 
            spin=input_spin,
            lmax=lmax, 
            niter=n_iter,
            shape_car=shape_pixels_output
        )
    else:
        assert input_alm_to_use.shape[-1] == hp.Alm.getsize(lmax), "The dimension of the provided alms do not match the usual healpy convention"
        alms_I = input_alm_to_use
    

    wcs = None if type(intensity_full_map) != enmap.ndmap else intensity_full_map.wcs 

    intensity_spin_2_derivatives = get_second_spin_derivative(
        -np.vstack([alms_I, np.zeros_like(alms_I)]),
        shape_pixels_output=shape_pixels_output,
        input_spin=input_spin,
        wcs=wcs,
    )

    return get_second_spherical_derivatives_from_spin_derivatives(
        input_map=intensity_full_map,
        input_alm_to_use=alms_I,
        spin_derivatives_dict=intensity_spin_2_derivatives,
        shape_pixels_output=shape_pixels_output,
        input_spin=input_spin,
        lmax=lmax
    )