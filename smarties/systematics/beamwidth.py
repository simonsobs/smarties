# This file is part of SMARTIES.
# Copyright (c) 2024-2026 bers of the Simons Simons Observatory Collaboration.
# lease refer to the LICENSE file in the root of this repository.

import numpy as np
import healpy as hp
from pixell import enmap
from opt_einsum import contract

from smarties.harmonics import map2alm_anypix
from smarties.hn import Spin_maps
from smarties.external.s4cmb import (
    get_first_spin_derivative, 
    get_second_spin_derivative,
    multiply_tan_theta_power
)

__all__ = [
    'create_beamwidth_spin_leakage_map_BICEP'
]

def _build_central_term_pointing_leakage(
        intensity_CMB,
        lmax=None,
        mask=None,
        spherical_derivatives=None,
        shape_car=None,
        shape_fullsky_car=None
    ):
    """
    Build the central term of the beamwidth leakage maps from T, corresponding
    to a partial Laplacian applied to the intensity CMB map 
    $$\nabla I$$
    
    Parameters
    ----------
    intensity_CMB: np.ndarray
        Intensity CMB map already convolved with Gaussian circularly-symmetric beam 
    lmax: int (optional)
        Maximum multipole for the computation of the spin derivatives of the 
        intensity CMB map (if spherical_derivatives is not provided)
    mask: np.ndarray (optional)
        Mask to apply to the maps, of the same dimension as the intensity_CMB map
        If output maps will be flattened and only containing the pixels where 
        the mask is not zero
    spherical_derivatives: dict (optional)
        Dictionary containing the spherical derivatives of the intensity CMB map, 
        must contain keys being 'theta_theta' and 'phi_phi' for the derivatives 
        with respect to theta and phi respectively (including the factor of 1/sin theta)
        If not provided, the spherical derivatives will be re-computed from the 
        provided intensity_CMB map.
    shape_car: tuple (optional)
        Shape of the cut masked CAR maps if the provided intensity_CMB map is in 
        CAR pixelization, needed to compute the spherical derivatives if not provided
    shape_fullsky_car:
        Shape of the full sky CAR maps if the provided intensity_CMB map is in 
        CAR pixelization, needed to compute the spherical derivatives if not provided

    Returns
    -------
    central_term: np.ndarray
        Central term of the beamwidth leakage maps, of shape (npix,) where npix is
        the number of pixels in the output maps (after masking if mask is provided)
        

    """

    if mask is None:
        mask_bool = ...
    else:
        mask_bool = ..., mask != 0

    if spherical_derivatives is None:
        print("Computing spherical derivatives from the temperature map", flush=True)

        if type(intensity_CMB) == np.ndarray:
            # HEALPIX pixelization expected
            assert intensity_CMB.ndim == 1, 'The intensity_CMB map must be a 1D array compatible with a full sky healpy map'
            assert np.log(np.sqrt(intensity_CMB.size/12)) / np.log(2) % 1 == 0, 'The intensity_CMB map dimension must be compatible with a full sky healpy map'
            shape_car = None
            take_box_function = lambda x: x
        elif type(intensity_CMB) == enmap.ndmap:
            # CAR pixelization expected
            assert shape_car is not None, 'The shape_car must be provided if the intensity map is provided as enmap.ndmap'
            
            if shape_fullsky_car is None:
                shape_fullsky_car = intensity_CMB.shape if intensity_CMB.ndim == 2 else None
            assert shape_fullsky_car is not None

            assert type(mask) == enmap.ndmap, 'The provided mask must be of type ndmap if the intensity map is as well provided with this type'
            box_coordinates = mask.reshape(shape_car).corners()

            take_box_function = lambda x: x.reshape(shape_fullsky_car).submap(box_coordinates).ravel()

        else:
            raise ValueError("The intensity_CMB map must be either a 1D array compatible with a full sky healpy map or a 2D array compatible with a CAR array")

        alms_I = map2alm_anypix(
            intensity_CMB,
            spin=0,
            lmax=lmax, 
            niter=10,
            shape_car=shape_fullsky_car
        )

        input_spin = 0
        intensity_spin_2_derivatives = get_second_spin_derivative(
            -np.vstack([alms_I, np.zeros_like(alms_I)]),
            shape_pixels_output=(intensity_CMB.size,),
            input_spin=input_spin,
        )
        intensity_spin_1_derivatives = get_first_spin_derivative(
            -np.vstack([alms_I, np.zeros_like(alms_I)]),
            shape_pixels_output=(intensity_CMB.size,),
            input_spin=input_spin,
        )
        
        
        central_term = take_box_function(
            intensity_spin_2_derivatives['+1-1'],
        )[mask_bool] - take_box_function(
            multiply_tan_theta_power(
                -0.5 * (
                    intensity_spin_1_derivatives[input_spin+1] 
                    + intensity_spin_1_derivatives[input_spin-1]
                ), # d/dtheta I
                power=-1,
                shape_car=shape_car
                )
        )[mask_bool]

    else:
        print("Using provided spherical derivatives", flush=True)
        assert 'theta_theta' in spherical_derivatives, "The provided spherical_derivatives dictionnary must contain the 'theta_theta' key for the second derivative with respect to theta"
        assert 'phi_phi' in spherical_derivatives, "The provided spherical_derivatives dictionnary must contain the 'phi_phi' key for the second derivative with respect to phi"
        
        print("Computing central term ...", flush=True)
        if shape_car is not None:
            for spin in spherical_derivatives:
                if spherical_derivatives[spin].shape != np.prod(shape_car):
                    raise ValueError(f"The provided spherical_derivatives for spin {spin} have a shape {spherical_derivatives[spin].shape} that is not compatible with the expected shape of the flattened CAR maps {np.prod(shape_car)}")

        central_term = (
            spherical_derivatives['theta_theta'][mask_bool] 
            + spherical_derivatives['phi_phi'][mask_bool]
        )

    return central_term


def create_beamwidth_spin_leakage_map_BICEP(
        intensity_CMB, 
        delta_sigma,
        sigma_fwhm,
        lmax=None,
        mask=None,
        spherical_derivatives=None,
        shape_car=None,
        shape_fullsky_car=None
    ):
    """
    Get the differential beamwidth maps for a given intensity CMB map and 
    differential pointing parameters using BICEP templates as described in the 
    formalism described in BICEP2 III: INSTRUMENTAL SYSTEMATICS, arXiv:1502.00608, 
    with the derivatives being computed directly from the spherical derivatives of 
    the provided intensity CMB map. 

    The x and y directions are assumed to be respectively along -theta and -phi
    at a rest position of the detector, and rotated when scanned. 

    Parameters
    ----------
    intensity_CMB: np.ndarray
        Intensity CMB map already convolved with Gaussian circularly-symmetric 
        beam
    delta_sigma: np.ndarray | float
        Angular fwhm mismatch for each detector in radians 
    lmax: int (optional)
        Maximum multipole for the computation of the spin derivatives of the 
        intensity CMB map if spherical_derivatives is not provided
    mask: np.ndarray (optional)
        Mask to apply to the maps, of the same dimension as the intensity_CMB map
        If output maps will be flattened and only containing the pixels where 
        the mask is not zero 
    spherical_derivatives: dict (optional)
        Dictionary containing the spherical derivatives of the intensity CMB map, 
        must contain keys being 'theta' and 'phi' for the derivatives with respect 
        to theta and phi respectively (including the factor of 1/sin theta)
        If not provided, the spherical derivatives will be re-computed from the 
        provided intensity_CMB map.
    shape_car: tuple (optional)
        Shape of the cut masked CAR maps if the provided intensity_CMB map is 
        in CAR pixelization
    shape_fullsky_car: tuple (optional)
        Shape of the full sky CAR maps if the provided intensity_CMB map is 
        in CAR pixelization, needed to compute the spherical derivatives

    Returns
    -------
    pointing_leakage_spin_maps: dictionary 
        Dictionary of pointing leakage maps, each of shape (npix,), with keys 
        being spin=1 and -1 

    Note
    ----
    Only the temperature leakage is considered here, the polarization leakage 
    is not implemented. 
    """

    delta_sigma = np.asarray(delta_sigma)

    assert np.array(delta_sigma).ndim == 1, 'The dimension of the delta_sigma must be (n_det,)'

    sigma_cs = np.asarray(sigma_fwhm) / ((8 * np.log(2)) ** 0.5) * np.pi/(180*60)  
    # convert from FWHM to sigma_cs, in radians


    central_term = _build_central_term_pointing_leakage(
        intensity_CMB,
        lmax=lmax,
        mask=mask,
        spherical_derivatives=spherical_derivatives,
        shape_car=shape_car,
        shape_fullsky_car=shape_fullsky_car
    )

    beamwidth_leakage_spin_maps = Spin_maps()

    # Spin 0
    beamwidth_leakage_spin_maps[0] = contract(
        'd,p->dp', 
        sigma_cs * delta_sigma, 
        central_term
    )

    return beamwidth_leakage_spin_maps
