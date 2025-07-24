# This file is part of SMARTIES.
# Copyright (C) 2024 CNRS / SciPol developers
#
# SMARTIES is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SMARTIES is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SMARTIES. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import healpy as hp
from opt_einsum import contract

from smarties.hn import Spin_maps
from smarties.external.s4cmb import get_second_spin_derivative
from smarties.tools import get_rotation_matrix

def get_ellipse_deviation(ellipticity, sigma_cs):
    """
    Get the ellipticity deviation $\Delta_\sigma$ from the input ellipticity (third eccentricity) given by:
        $$ ellipticity = (\sigma_{\rm maj}^2 - \sigma_{\rm min}^2) / (\sigma_{\rm maj}^2 + \sigma_{\rm min}^2) $$
    where $\sigma_{\rm maj}$ and $\sigma_{\rm min}$ are the major and minor axes of the ellipse, respectively, and
    defined as:
        $$ \sigma_{\rm maj} = \sigma_{\rm cs} + \Delta_\sigma / 2 $$.
        $$ \sigma_{\rm min} = \sigma_{\rm cs} - \Delta_\sigma / 2 $$.
    
    Parameters
    ----------
    ellipticity: np.ndarray
        Ellipticity parameter for each detector
    sigma_cs: np.ndarray
        Circularly-symmetric beam width for each detector, in arcmin

    Returns
    -------
    delta_sigma: np.ndarray
        Ellipticity deviation $\Delta_\sigma$ for each detector so that the major and minor axes of the ellipse are given by:
        $$ \sigma_{\rm maj} = \sigma_{\rm cs} + \Delta_\sigma / 2 $$
        $$ \sigma_{\rm min} = \sigma_{\rm cs} - \Delta_\sigma / 2 $$
        without taking into account the rotation of the ellipse.

    """
    ellipticity = np.asarray(ellipticity)
    sigma_cs = np.asarray(sigma_cs)

    assert ellipticity.ndim == 1, 'The ellipticity map must have only 1 dimension'
    assert sigma_cs.shape == ellipticity.shape, 'The ellipticity and sigma_cs maps must have the same shape'

    

    return np.where(
        ellipticity != 0, 
        2 * sigma_cs * (1 - np.sqrt(1 - ellipticity ** 2)) / ellipticity,
        0
    ) # the formula is not defined for ellipticity = 0, which correspond to a circular beam where the deviation is 0


def get_differential_ellipticity(
        intensity_CMB,
        ellipticity,
        ellipse_angle,
        sigma_FWHM,
        lmax=None,
        mask=None,
        bool_secondary_term=True,
    ):
    """
    Get the differential ellipticity maps for a given intensity CMB map and ellipticity parameters as described in the formalism describe in arXiv:2011.13910, with output spins 0, 2, -2. 
    
    Parameters
    ----------
    intensity_CMB: np.ndarray
        Full sky intensity CMB map already convolved with Gaussian circularly-symmetric beam (as assumed in the formalism), the output maps will have the same dimension
    ellipticity: np.ndarray
        Ellipticity parameter provided for each detector, defined as the ratio of the difference between the squares of the major and minor axes of the ellipse to their sum, i.e. $\epsilon = (\sigma_{\rm maj}^2 - \sigma_{\rm min}^2) / (\sigma_{\rm maj}^2 + \sigma_{\rm min}^2)$, which is also two times the third eccentricity parameter
    ellipse_angle: np.ndarray
        Angle of the ellipse in radians, defined as the angle between the major axis and the x-axis, for each detector
    sigma_FWHM: np.ndarray
        Full width at half maximum of the beam in arcmin, for each detector, used to compute the circularly-symmetric (cs) beam width $\sigma_{\rm cs}$ as $\sigma_{\rm cs} = \frac{\rm FWHM}{\sqrt{8 \ln(2)}}$
    lmax: int
        Maximum multipole for the computation of the spin derivatives of the intensity CMB map
    mask: np.ndarray, optional
        HEALPix mask to define the area of the sky to compute the differential systematics maps. If None, the full sky is used.
    bool_secondary_term: bool, optional
        If False, ignore the secondary term in the differential ellipticity formalism.

    Returns
    -------
    differential_ellipticity_spin_maps: dictionary 
        Dictionary of differential ellipticity maps, each of shape (n_det,npix), with keys being spin=0, 2, -2 

    Notes
    -----
    Currently, the input intensity_CMB map is assumed to be a full sky map, i.e. it must have a dimension of 12 * nside^2, where nside is the HEALPix nside parameter, and smooth with the circularly-symmetric beam defined by the sigma_FWHM parameter. 
    """

    #TODO: Allow for intensity_CMB to be different for each detector in case sigma_FWHM is different for each detector, i.e. allow for a 2D array of shape (n_det, npix) for intensity_CMB 

    intensity_CMB = np.asarray(intensity_CMB)
    assert intensity_CMB.ndim == 1, 'The intensity_CMB map must have only 1 dimension'
    assert np.log(np.sqrt(intensity_CMB.size/12)) / np.log(2) % 1 == 0, 'The intensity_CMB map dimension must be compatible with a full sky healpy map'
    nside = hp.npix2nside(intensity_CMB.size)

    ellipticity = np.asarray(ellipticity)
    ellipse_angle = np.asarray(ellipse_angle)
    sigma_cs = np.asarray(sigma_FWHM) / ((8 * np.log(2)) ** 0.5) * np.pi/(180*60)  # convert from FWHM to sigma_cs, in radians

    assert ellipticity.ndim == 1, 'The ellipticity map must have only 1 dimension'
    assert ellipticity.shape == sigma_cs.shape, 'The ellipticity and sigma_cs maps must have the same shape'
    assert ellipticity.shape == ellipse_angle.shape, 'The ellipticity and ellipse angle maps must have the same shape'

    if mask is None:
        mask_bool = ...
    else:
        mask_bool = mask != 0

    if bool_secondary_term:
        coefficient_secondary_term = 1
    else:
        coefficient_secondary_term = 0
    
    alms_I = hp.map2alm(intensity_CMB, lmax=lmax, iter=10)

    intensity_spin_2_derivatives = get_second_spin_derivative(
        -np.vstack([alms_I.real, np.zeros_like(alms_I)]), 
        nside=nside,
        input_spin=0, 
    )

    rotation_matrix_ellipse_angle = get_rotation_matrix(ellipse_angle)
    delta_sigma = get_ellipse_deviation(ellipticity, sigma_cs)

    propagation_perturbation_ellipse = np.einsum('dxy, xd, dxa->dya',
                                                 rotation_matrix_ellipse_angle,
                                                 np.vstack([delta_sigma, -delta_sigma]),
                                                 rotation_matrix_ellipse_angle
                                                )


    alpha_2 = contract('d, dxy->dxy', 
                       sigma_cs**3 / (sigma_cs ** 2 - delta_sigma ** 2), 
                       propagation_perturbation_ellipse
                    ) - coefficient_secondary_term * np.broadcast_to(
                        delta_sigma ** 2 * sigma_cs ** 2 / ((sigma_cs ** 2 - delta_sigma ** 2)), 
                        (2, 2, sigma_cs.size)
                    ).T * np.eye(2)

    alpha_0 = sigma_cs**2 / (sigma_cs ** 2 - delta_sigma ** 2) + np.linalg.trace(alpha_2)/sigma_cs**2
    alpha_2 = contract('dxy,d->dxy', alpha_2, 1/alpha_0)

    differential_ellipticity_spin_maps = Spin_maps()

    # Spin 0
    print("Computing spin 0 differential ellipticity map ...", flush=True)
    differential_ellipticity_spin_maps[0] = contract('d,p->dp',  np.linalg.trace(alpha_2) / 2., intensity_spin_2_derivatives['+1-1'][mask_bool], memory_limit='max_input')

    spin_2_prefactor = (alpha_2[...,1,1] - alpha_2[...,0,0]) / 2. - 1j*(alpha_2[...,1,0] + alpha_2[...,0,1]) / 2.
    # Spin 2
    print("Computing spin -2 differential ellipticity map ...", flush=True)
    differential_ellipticity_spin_maps[2] = contract('d,p->dp', spin_2_prefactor, intensity_spin_2_derivatives[-2][mask_bool], memory_limit='max_input')

    # Spin -2
    print("Computing spin 2 differential ellipticity map ...", flush=True)
    differential_ellipticity_spin_maps[-2] = contract('d,p->dp', np.conj(spin_2_prefactor), intensity_spin_2_derivatives[2][mask_bool], memory_limit='max_input')

    return differential_ellipticity_spin_maps
