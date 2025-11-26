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
from smarties.external.s4cmb import (
    get_second_spin_derivative, 
    get_second_spherical_derivatives_from_spin_derivatives, 
    multiply_tan_theta_power
)
from smarties.tools import convert_ellipticities_conventions

def get_differential_ellipticity_BICEP_TOAST(
        intensity_CMB,
        ellipticity_parameters_dict,
        sigma_FWHM,
        lmax=None,
        mask=None,
        spherical_derivatives=None,
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
    ellipticity_parameter_convention: str, optional
        Convention used for the ellipticity parameter, either 'Third flattening' or 'Third eccentricity', default is 'Third flattening'
    Returns
    -------
    differential_ellipticity_spin_maps: dictionary 
        Dictionary of differential ellipticity maps, each of shape (n_det,npix), with keys being spin=0, 2, -2 

    Notes
    -----
    Currently, the input intensity_CMB map is assumed to be a full sky map, i.e. it must have a dimension of 12 * nside^2, where nside is the HEALPix nside parameter, and smooth with the circularly-symmetric beam defined by the sigma_FWHM parameter. 
    """

    assert 'ellipticity_parameter_convention' in ellipticity_parameters_dict or ('dp' in ellipticity_parameters_dict and 'dc' in ellipticity_parameters_dict), "The provided ellipticity_parameters_dict must contain the 'ellipticity_parameter_convention' key to specify the ellipticity parameter convention used to compute the dp and dc parameters"
    if 'ellipticity_parameter_convention' not in ellipticity_parameters_dict:
        ellipticity_parameters_dict['ellipticity_parameter_convention'] = 'Plus-Cross ellipticity'

    p_c_parameters_dict = convert_ellipticities_conventions(
        ellipticity_parameters_dict,
        sigma_FWHM=sigma_FWHM,
        input_ellipticity_convention=ellipticity_parameters_dict['ellipticity_parameter_convention'],
        output_ellipticity_convention='Plus-Cross ellipticity',
    )

    parameter_p = p_c_parameters_dict['dp']
    parameter_c = p_c_parameters_dict['dc']
    assert parameter_p.ndim == 1, 'The parameter p provided must have shape (n_det)'
    assert parameter_c.ndim == 1, 'The parameter c provided must have shape (n_det)'
    assert parameter_p.shape == parameter_c.shape, 'The parameter p and c provided must have the same shape'

    intensity_CMB = np.asarray(intensity_CMB)
    assert intensity_CMB.ndim == 1, 'The intensity_CMB map must have only 1 dimension'
    assert np.log(np.sqrt(intensity_CMB.size/12)) / np.log(2) % 1 == 0, 'The intensity_CMB map dimension must be compatible with a full sky healpy map'
    nside = hp.npix2nside(intensity_CMB.size)

    sigma_cs = np.asarray(sigma_FWHM) / ((8 * np.log(2)) ** 0.5) * np.pi/(180*60)  # convert from FWHM to sigma_cs, in radians

    if mask is None:
        mask_bool = ...
    else:
        mask_bool = mask != 0
    
    if spherical_derivatives is None:
        print("Computing spherical derivatives from the temperature map", flush=True)
        alms_I = hp.map2alm(intensity_CMB, lmax=lmax, iter=10)

        input_spin = 0
        intensity_spin_2_derivatives = get_second_spin_derivative(
            -np.vstack([alms_I, np.zeros_like(alms_I)]),
            nside=nside,
            input_spin=input_spin,
        )
        spherical_derivatives = get_second_spherical_derivatives_from_spin_derivatives(
            input_map=intensity_CMB,
            spin_derivatives_dict=intensity_spin_2_derivatives,
            nside=nside,
            lmax=lmax,
            input_spin=input_spin
        )
        
        central_term = (
            (intensity_spin_2_derivatives[input_spin+2]  
            - intensity_spin_2_derivatives[input_spin-2])/(4*1j)
            + spherical_derivatives['theta_phi']
        )[mask_bool]
    else:
        print("Using provided spherical derivatives", flush=True)
        assert 'phi_phi' in spherical_derivatives, "The provided spherical_derivatives dictionnary must contain the 'phi_phi' key for the second derivative with respect to phi (including factor 1/sin^2(theta))"
        assert 'theta_phi' in spherical_derivatives, "The provided spherical_derivatives dictionnary must contain the 'theta_phi' key for the second derivative with respect to theta and phi"
        assert 'theta_theta' in spherical_derivatives, "The provided spherical_derivatives dictionnary must contain the 'theta_theta' key for the second derivative with respect to theta"
        assert 'phi' in spherical_derivatives, "The provided spherical_derivatives dictionnary must contain the 'phi' key for the first derivative with respect to phi"

        central_term = (2 * spherical_derivatives['theta_phi']
                        - multiply_tan_theta_power(
                            spherical_derivatives['phi'],
                            nside,
                            power=-1
                        )
                    )[mask_bool]

    differential_ellipticity_spin_maps = Spin_maps()

    # Spin 0
    print("Computing spin 0 differential ellipticity map ...", flush=True)
    # differential_ellipticity_spin_maps[0] = contract('d,p->dp',  np.linalg.trace(alpha_2) / 2., intensity_spin_2_derivatives['+1-1'][mask_bool], memory_limit='max_input')

    # Spin 2
    print("Computing spin -2 differential ellipticity map ...", flush=True)

    differential_ellipticity_spin_maps[-2] = contract(
        'd,p->dp', 
        sigma_cs**2 /2 * (1j*parameter_p + parameter_c),
        central_term - 1j * spherical_derivatives['phi_phi'][mask_bool], 
        memory_limit='max_input'
    ) + contract(
        'd,p->dp', 
        sigma_cs**2 /2 * ( parameter_p + 1j * parameter_c ),
        spherical_derivatives['theta_theta'][mask_bool], 
        memory_limit='max_input'
    )

    # Spin -2
    print("Computing spin 2 differential ellipticity map ...", flush=True)
    differential_ellipticity_spin_maps[2] = contract(
        'd,p->dp', 
        sigma_cs**2 /2 * (-1j*parameter_p + parameter_c),
        central_term + 1j * spherical_derivatives['phi_phi'][mask_bool], 
        memory_limit='max_input'
    ) + contract(
        'd,p->dp', 
        sigma_cs**2 /2 * ( parameter_p - 1j * parameter_c ),
        spherical_derivatives['theta_theta'][mask_bool], 
        memory_limit='max_input'
    )

    differential_ellipticity_spin_maps[0] = np.zeros_like(differential_ellipticity_spin_maps[2])

    return differential_ellipticity_spin_maps
