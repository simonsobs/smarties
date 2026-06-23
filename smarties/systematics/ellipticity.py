# This file is part of SMARTIES.
# Copyright (c) 2024-2026 bers of the Simons Simons Observatory Collaboration.
# lease refer to the LICENSE file in the root of this repository.

import numpy as np
import healpy as hp
from pixell import enmap
from opt_einsum import contract

from smarties.utils.harmonics import map2alm_anypix
from smarties.hn import Spin_maps
from smarties.external.s4cmb import (
    get_first_spin_derivative,
    get_second_spin_derivative, 
    get_second_spherical_derivatives_from_spin_derivatives, 
    multiply_tan_theta_power
)
from smarties.utils.tools import convert_ellipticities_conventions, get_rotation_matrix

__all__ = [
    'get_differential_ellipticity_BICEP',
    'get_differential_ellipticity_no_calibration'
]

def get_differential_ellipticity_BICEP(
        intensity_CMB,
        ellipticity_parameters_dict,
        sigma_fwhm,
        lmax=None,
        mask=None,
        spherical_derivatives=None,
        shape_car=None,
        shape_fullsky_car=None
    ):
    """
    Get the differential ellipticity maps for a given intensity CMB map and 
    ellipticity parameters using BICEP templates as described in the formalism 
    described in BICEP2 III: INSTRUMENTAL SYSTEMATICS, arXiv:1502.00608, with 
    the derivatives being computed directly from the spherical derivatives of 
    the provided intensity CMB map. 
    
    Parameters
    ----------
    intensity_CMB: np.ndarray | enmap.ndmap
        Full sky intensity CMB map assumed to be already convolved with Gaussian 
        circularly-symmetric beam of the parameter sigma_FWHP, the output maps 
        will have the same dimension
    ellipticity: dict
        Dictionary of the ellipticity parameters provided for each detector, see
        must contain keys 'ellipticity_parameter_convention', see the function
        `convert_ellipticities_conventions` in `tools.py` for more details
    sigma_fwhm: np.ndarray | float
        Full width at half maximum of the beam in arcmin, provided for each detector
        or as a float, assumed to have been used to compute the circularly-symmetric 
        beam smoothing the intensity_CMB map, and around which the Taylor expansion 
        of the beam ellipticity is performed in the BICEP formalism
    lmax: int (optional)
        Maximum multipole for the computation of the spin derivatives of the intensity CMB map
    mask: np.ndarray (optional)
        Mask to define the area of the sky to compute the differential systematics maps. 
        If None, the full sky is used.
    spherical_derivatives: dict (optional)
        Dictionary of the spherical derivatives of the intensity CMB map, 
        with keys 'phi_phi', 'theta_phi', 'theta_theta' for the second derivatives, 
        and 'phi' for the first derivative with respect to phi, including the 
        factor 1/sin^2(theta) for the derivatives with respect to phi. If not 
        provided, the spherical derivatives will be computed from the intensity 
        CMB map using the spin-derivative formalism. If provided, they will be 
        used directly to compute the differential ellipticity maps, which can 
        save time if multiple systematics maps are computed from the same intensity CMB map.
    Returns
    -------
    differential_ellipticity_spin_maps: dictionary 
        Dictionary of differential ellipticity maps, each of shape (n_det,npix), 
        with keys being spin=0, 2, -2 

    Notes
    -----
    Currently, the input intensity_CMB map is assumed to be a full sky map, 
    i.e. it must have a dimension of 12 * nside^2 or the full sky car dimension, 
    and smoothed with the circularly-symmetric beam defined by the sigma_fwhm parameter. 
    """

    assert 'ellipticity_parameter_convention' in ellipticity_parameters_dict or ('dp' in ellipticity_parameters_dict and 'dc' in ellipticity_parameters_dict), "The provided ellipticity_parameters_dict must contain the 'ellipticity_parameter_convention' key to specify the ellipticity parameter convention used to compute the dp and dc parameters"
    if 'ellipticity_parameter_convention' not in ellipticity_parameters_dict:
        ellipticity_parameters_dict['ellipticity_parameter_convention'] = 'Plus-Cross ellipticity'

    p_c_parameters_dict = convert_ellipticities_conventions(
        ellipticity_parameters_dict,
        sigma_fwhm=sigma_fwhm,
        input_ellipticity_convention=ellipticity_parameters_dict['ellipticity_parameter_convention'],
        output_ellipticity_convention='Plus-Cross ellipticity',
    )

    parameter_p = p_c_parameters_dict['dp']
    parameter_c = p_c_parameters_dict['dc']
    assert parameter_p.ndim == 1, 'The parameter p provided must have shape (n_det)'
    assert parameter_c.ndim == 1, 'The parameter c provided must have shape (n_det)'
    assert parameter_p.shape == parameter_c.shape, 'The parameter p and c provided must have the same shape'

    if mask is None:
        mask_bool = ...
    else:
        mask_bool = ..., mask != 0
    
    take_box_function = lambda x: x#[mask_bool]

    
    sigma_cs = np.asarray(sigma_fwhm) / ((8 * np.log(2)) ** 0.5) * np.pi/(180*60)  # convert from FWHM to sigma_cs, in radians
    if spherical_derivatives is None:
        print("Computing spherical derivatives from the temperature map", flush=True)

        if type(intensity_CMB) == np.ndarray:
            # HEALPIX pixelization expected
            assert intensity_CMB.ndim == 1, 'The intensity_CMB map must be a 1D array compatible with a full sky healpy map'
            assert np.log(np.sqrt(intensity_CMB.size/12)) / np.log(2) % 1 == 0, 'The intensity_CMB map dimension must be compatible with a full sky healpy map'
            shape_car = None
        elif type(intensity_CMB) == enmap.ndmap:
            # CAR pixelization expected
            assert shape_car is not None, 'The shape_car must be provided if the intensity map is provided as enmap.ndmap'
            
            if shape_fullsky_car is None:
                shape_fullsky_car = intensity_CMB.shape if intensity_CMB.ndim == 2 else None
            assert shape_fullsky_car is not None

            assert type(mask) == enmap.ndmap, 'The provided mask must be of type ndmap if the intensity map is as well provided with this type'
            box_coordinates = mask.reshape(shape_car).corners()

            take_box_function = lambda x: x.reshape(shape_fullsky_car).submap(box_coordinates).ravel()#[mask_bool]

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
        intensity_spin_1_derivatives = get_first_spin_derivative(
            -np.vstack([alms_I, np.zeros_like(alms_I)]),
            shape_pixels_output=(intensity_CMB.size,),
            input_spin=input_spin,
        )
        intensity_spin_2_derivatives = get_second_spin_derivative(
            -np.vstack([alms_I, np.zeros_like(alms_I)]),
            shape_pixels_output=(intensity_CMB.size,),
            input_spin=input_spin,
        )
        
        
        central_term = take_box_function(
            (intensity_spin_2_derivatives[input_spin+2]  
            - multiply_tan_theta_power(
                intensity_spin_1_derivatives[input_spin+1],
                power=-1,
                shape_car=shape_car
                )
            )
        )[mask_bool]

    else:
        print("Using provided spherical derivatives", flush=True)
        assert 'phi_phi' in spherical_derivatives, "The provided spherical_derivatives dictionnary must contain the 'phi_phi' key for the second derivative with respect to phi (including factor 1/sin^2(theta))"
        assert 'theta_phi' in spherical_derivatives, "The provided spherical_derivatives dictionnary must contain the 'theta_phi' key for the second derivative with respect to theta and phi"
        assert 'theta_theta' in spherical_derivatives, "The provided spherical_derivatives dictionnary must contain the 'theta_theta' key for the second derivative with respect to theta"
        assert 'phi' in spherical_derivatives, "The provided spherical_derivatives dictionnary must contain the 'phi' key for the first derivative with respect to phi"
        
        print("Computing central term ...", flush=True)
        for spin in spherical_derivatives:
            if spherical_derivatives[spin].shape != np.prod(shape_car):
                spherical_derivatives[spin] = take_box_function(spherical_derivatives[spin])
            

        central_term = (
            spherical_derivatives['phi_phi'][mask_bool] 
            - spherical_derivatives['theta_theta'][mask_bool]
        ) - 1j*(multiply_tan_theta_power(
                spherical_derivatives['phi'],
                power=-1,
                shape_car=shape_car
            )[mask_bool]  
            - 2 * spherical_derivatives['theta_phi'][mask_bool] 
        )

    differential_ellipticity_spin_maps = Spin_maps()

    # Spin -2
    print("Computing spin -2 differential ellipticity map ...", flush=True)

    differential_ellipticity_spin_maps[-2] = contract(
        'd,...->d...', 
        sigma_cs**2 /2 * (parameter_p - 1j*parameter_c),
        central_term, 
        memory_limit='max_input'
    ) 

    # Spin 2
    print("Computing spin 2 differential ellipticity map ...", flush=True)
    differential_ellipticity_spin_maps[2] = np.conj(
        differential_ellipticity_spin_maps[-2]
    )

    differential_ellipticity_spin_maps[0] = np.zeros_like(
        differential_ellipticity_spin_maps[-2]
    )

    if type(intensity_CMB) == enmap.ndmap:
        for spin in differential_ellipticity_spin_maps.spins:
            differential_ellipticity_spin_maps[spin] = enmap.ndmap(
                differential_ellipticity_spin_maps[spin],
                wcs=intensity_CMB.wcs
            )
    else:
        for spin in differential_ellipticity_spin_maps.spins:
            differential_ellipticity_spin_maps[spin] = differential_ellipticity_spin_maps[spin].squeeze()

    return differential_ellipticity_spin_maps

def old_get_differential_ellipticity_BICEP(
        intensity_CMB,
        ellipticity_parameters_dict,
        sigma_fwhm,
        lmax=None,
        mask=None,
        spherical_derivatives=None,
        shape_car=None,
        shape_fullsky_car=None
    ):
    """
    Get the differential ellipticity maps for a given intensity CMB map and 
    ellipticity parameters using BICEP templates as described in the formalism 
    described in BICEP2 III: INSTRUMENTAL SYSTEMATICS, arXiv:1502.00608, with 
    the derivatives being computed directly from the spherical derivatives of 
    the provided intensity CMB map. 

    **WARNNING** 
    Old version of the function with wrong templates, should not be used, 
    kept here for reference and comparison with the new version of the function 
    `get_differential_ellipticity_BICEP` in which the correct templates are implemented.
    Will be removed in future versions of the code. 

    
    Parameters
    ----------
    intensity_CMB: np.ndarray | enmap.ndmap
        Full sky intensity CMB map assumed to be already convolved with Gaussian 
        circularly-symmetric beam of the parameter sigma_FWHP, the output maps 
        will have the same dimension
    ellipticity: dict
        Dictionary of the ellipticity parameters provided for each detector, see
        must contain keys 'ellipticity_parameter_convention', see the function
        `convert_ellipticities_conventions` in `tools.py` for more details
    sigma_fwhm: np.ndarray | float
        Full width at half maximum of the beam in arcmin, provided for each detector
        or as a float, assumed to have been used to compute the circularly-symmetric 
        beam smoothing the intensity_CMB map, and around which the Taylor expansion 
        of the beam ellipticity is performed in the BICEP formalism
    lmax: int (optional)
        Maximum multipole for the computation of the spin derivatives of the intensity CMB map
    mask: np.ndarray (optional)
        Mask to define the area of the sky to compute the differential systematics maps. 
        If None, the full sky is used.
    spherical_derivatives: dict (optional)
        Dictionary of the spherical derivatives of the intensity CMB map, 
        with keys 'phi_phi', 'theta_phi', 'theta_theta' for the second derivatives, 
        and 'phi' for the first derivative with respect to phi, including the 
        factor 1/sin^2(theta) for the derivatives with respect to phi. If not 
        provided, the spherical derivatives will be computed from the intensity 
        CMB map using the spin-derivative formalism. If provided, they will be 
        used directly to compute the differential ellipticity maps, which can 
        save time if multiple systematics maps are computed from the same intensity CMB map.
    Returns
    -------
    differential_ellipticity_spin_maps: dictionary 
        Dictionary of differential ellipticity maps, each of shape (n_det,npix), 
        with keys being spin=2, -2 

    Notes
    -----
    Currently, the input intensity_CMB map is assumed to be a full sky map, 
    i.e. it must have a dimension of 12 * nside^2 or the full sky car dimension, 
    and smoothed with the circularly-symmetric beam defined by the sigma_fwhm parameter. 
    """

    assert 'ellipticity_parameter_convention' in ellipticity_parameters_dict or (
        'dp' in ellipticity_parameters_dict and 'dc' in ellipticity_parameters_dict
    ), "The provided ellipticity_parameters_dict must contain the 'ellipticity_parameter_convention' key to specify the ellipticity parameter convention used to compute the dp and dc parameters"
    if 'ellipticity_parameter_convention' not in ellipticity_parameters_dict:
        ellipticity_parameters_dict['ellipticity_parameter_convention'] = 'Plus-Cross ellipticity'

    p_c_parameters_dict = convert_ellipticities_conventions(
        ellipticity_parameters_dict,
        sigma_fwhm=sigma_fwhm,
        input_ellipticity_convention=ellipticity_parameters_dict['ellipticity_parameter_convention'],
        output_ellipticity_convention='Plus-Cross ellipticity',
    )

    parameter_p = p_c_parameters_dict['dp']
    parameter_c = p_c_parameters_dict['dc']
    assert parameter_p.ndim == 1, 'The parameter p provided must have shape (n_det)'
    assert parameter_c.ndim == 1, 'The parameter c provided must have shape (n_det)'
    assert parameter_p.shape == parameter_c.shape, 'The parameter p and c provided must have the same shape'

    if mask is None:
        mask_bool = ...
    else:
        mask_bool = ..., mask != 0
    
    take_box_function = lambda x: x#[mask_bool]

    
    sigma_cs = np.asarray(sigma_fwhm) / ((8 * np.log(2)) ** 0.5) * np.pi/(180*60)  # convert from FWHM to sigma_cs, in radians
    if spherical_derivatives is None:
        print("Computing spherical derivatives from the temperature map", flush=True)

        if type(intensity_CMB) == np.ndarray:
            # HEALPIX pixelization expected
            assert intensity_CMB.ndim == 1, 'The intensity_CMB map must be a 1D array compatible with a full sky healpy map'
            assert np.log(np.sqrt(intensity_CMB.size/12)) / np.log(2) % 1 == 0, 'The intensity_CMB map dimension must be compatible with a full sky healpy map'
            shape_car = None
        elif type(intensity_CMB) == enmap.ndmap:
            # CAR pixelization expected
            assert shape_car is not None, 'The shape_car must be provided if the intensity map is provided as enmap.ndmap'
            
            if shape_fullsky_car is None:
                shape_fullsky_car = intensity_CMB.shape if intensity_CMB.ndim == 2 else None
            assert shape_fullsky_car is not None

            assert type(mask) == enmap.ndmap, 'The provided mask must be of type ndmap if the intensity map is as well provided with this type'
            box_coordinates = mask.reshape(shape_car).corners()

            take_box_function = lambda x: x.reshape(shape_fullsky_car).submap(box_coordinates).ravel()#[mask_bool]

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
        spherical_derivatives = get_second_spherical_derivatives_from_spin_derivatives(
            input_map=intensity_CMB,
            spin_derivatives_dict=intensity_spin_2_derivatives,
            shape_pixels_output=(intensity_CMB.size,),
            lmax=lmax,
            input_spin=input_spin
        )
        
        central_term = take_box_function(
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
        
        print("Computing central term ...", flush=True)
        for spin in spherical_derivatives:
            if spherical_derivatives[spin].shape != np.prod(shape_car):
                spherical_derivatives[spin] = take_box_function(spherical_derivatives[spin])
            

        central_term = 2 * spherical_derivatives['theta_phi'][mask_bool] - multiply_tan_theta_power(
                spherical_derivatives['phi'],
                power=-1,
                shape_car=shape_car
            )[mask_bool]

    differential_ellipticity_spin_maps = Spin_maps()

    # Spin 2
    print("Computing spin -2 differential ellipticity map ...", flush=True)

    differential_ellipticity_spin_maps[-2] = contract(
        'd,...->d...', 
        sigma_cs**2 /2 * (1j*parameter_p + parameter_c),
        central_term - 1j * spherical_derivatives['phi_phi'][mask_bool], 
        memory_limit='max_input'
    ) + contract(
        'd,...->d...', 
        sigma_cs**2 /2 * ( parameter_p + 1j * parameter_c ),
        spherical_derivatives['theta_theta'][mask_bool], 
        memory_limit='max_input'
    )

    # Spin -2
    print("Computing spin 2 differential ellipticity map ...", flush=True)
    differential_ellipticity_spin_maps[2] = contract(
        'd,...->d...', 
        sigma_cs**2 /2 * (-1j*parameter_p + parameter_c),
        central_term + 1j * spherical_derivatives['phi_phi'][mask_bool], 
        memory_limit='max_input'
    ) + contract(
        'd,...->d...', 
        sigma_cs**2 /2 * ( parameter_p - 1j * parameter_c ),
        spherical_derivatives['theta_theta'][mask_bool], 
        memory_limit='max_input'
    )

    differential_ellipticity_spin_maps[0] = np.zeros_like(differential_ellipticity_spin_maps[2])

    if type(intensity_CMB) == enmap.ndmap:
        for spin in differential_ellipticity_spin_maps.spins:
            differential_ellipticity_spin_maps[spin] = enmap.ndmap(
                differential_ellipticity_spin_maps[spin],
                wcs=intensity_CMB.wcs
            )
    else:
        for spin in differential_ellipticity_spin_maps.spins:
            differential_ellipticity_spin_maps[spin] = differential_ellipticity_spin_maps[spin].squeeze()

    return differential_ellipticity_spin_maps

def get_differential_ellipticity_no_calibration(
        intensity_CMB,
        ellipticity_parameters_dict,
        sigma_fwhm,
        lmax=None,
        mask=None,
        spherical_derivatives=None,
        shape_car=None,
        shape_fullsky_car=None,
        bool_secondary_term=False
    ):
    """
    Get the differential ellipticity maps for a given intensity CMB map and ellipticity parameters as described in the formalism describe in arXiv:2011.13910, with output spins 0, 2, -2. 
    
    Parameters
    ----------
    intensity_CMB: np.ndarray | enmap.ndmap
        Full sky intensity CMB map already convolved with Gaussian circularly-symmetric beam (as assumed in the formalism), the output maps will have the same dimension
    ellipticity: np.ndarray
        Ellipticity parameter provided for each detector, defined as the ratio of the difference between the squares of the major and minor axes of the ellipse to their sum, i.e. $\epsilon = (\sigma_{\rm maj}^2 - \sigma_{\rm min}^2) / (\sigma_{\rm maj}^2 + \sigma_{\rm min}^2)$, which is also two times the third eccentricity parameter
    ellipse_angle: np.ndarray
        Angle of the ellipse in radians, defined as the angle between the major axis and the x-axis, for each detector
    sigma_fwhm: np.ndarray
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
    Currently, the input intensity_CMB map is assumed to be a full sky map, i.e. it must have a dimension of 12 * nside^2, where nside is the HEALPix nside parameter, and smooth with the circularly-symmetric beam defined by the sigma_fwhm parameter. 
    """

    assert 'ellipticity_parameter_convention' in ellipticity_parameters_dict or ('dp' in ellipticity_parameters_dict and 'dc' in ellipticity_parameters_dict), "The provided ellipticity_parameters_dict must contain the 'ellipticity_parameter_convention' key to specify the ellipticity parameter convention used to compute the dp and dc parameters"
    if 'ellipticity_parameter_convention' not in ellipticity_parameters_dict:
        ellipticity_parameters_dict['ellipticity_parameter_convention'] = 'Plus-Cross ellipticity'

    sigma_cs = np.asarray(sigma_fwhm) * np.pi/(180*60) / ((8 * np.log(2)) ** 0.5) 
    
    parameters_elliptical = convert_ellipticities_conventions(
        ellipticity_parameters_dict,
        sigma_fwhm=sigma_fwhm,
        input_ellipticity_convention=ellipticity_parameters_dict['ellipticity_parameter_convention'],
        output_ellipticity_convention='Third flattening',
    )


    delta_sigma = parameters_elliptical['ellipticity_value'] * sigma_cs
    ellipticity_angle = parameters_elliptical['ellipticity_angle']
    assert delta_sigma.ndim == 1, 'The parameter delta_sigma rebuilt provided must have shape (n_det)'
    assert ellipticity_angle.ndim == 1, 'The parameter ellipticity_angle provided must have shape (n_det)'
    assert delta_sigma.shape == ellipticity_angle.shape, 'The parameter p and c provided must have the same shape'

    # if bool_secondary_term:
    #     coefficient_secondary_term = 1
    # else:
    #     coefficient_secondary_term = 0

    rotation_matrix_ellipse_angle = get_rotation_matrix(ellipticity_angle)
    propagation_perturbation_ellipse = np.einsum('dxy, xz, dza->dya',
                                                 rotation_matrix_ellipse_angle,
                                                 np.diag([1, -1]),
                                                 rotation_matrix_ellipse_angle
                                                )
    
    ratio_term = delta_sigma / sigma_cs
    prefactor = 1 / (sigma_cs ** 2 - delta_sigma ** 2)
    alpha_2 = 0.5 * (np.broadcast_to(
                        sigma_cs ** 4 * ratio_term ** 2 * (ratio_term ** 2 - 3) / (ratio_term ** 2 - 1)**2 * prefactor, 
                        (2, 2, sigma_cs.size)
                    ).T * np.eye(2) + contract(
                        'd, dxy->dxy', 
                        sigma_cs ** 4 * 2 * ratio_term / (ratio_term ** 2 - 1)**2  * prefactor, 
                        propagation_perturbation_ellipse
                    )
    )

    alpha_0 = 0.5 * (sigma_cs**2 + np.linalg.trace(alpha_2)) * prefactor

    if mask is None:
        mask_bool = ...
    else:
        mask_bool = ..., mask != 0
    
    npix = mask[mask_bool].size

    take_box_function = lambda x: x #[mask_bool]

    
    if spherical_derivatives is None:
        print("Computing spherical derivatives from the temperature map", flush=True)

        if type(intensity_CMB) == np.ndarray:
            # HEALPIX pixelization expected
            assert intensity_CMB.ndim == 1, 'The intensity_CMB map must be a 1D array compatible with a full sky healpy map'
            assert np.log(np.sqrt(intensity_CMB.size/12)) / np.log(2) % 1 == 0, 'The intensity_CMB map dimension must be compatible with a full sky healpy map'
            nside = hp.npix2nside(intensity_CMB.size)
            shape_car = None
        elif type(intensity_CMB) == enmap.ndmap:
            # CAR pixelization expected
            assert shape_car is not None, 'The shape_car must be provided if the intensity map is provided as enmap.ndmap'
            
            nside = None
            if shape_fullsky_car is None:
                shape_fullsky_car = intensity_CMB.shape if intensity_CMB.ndim == 2 else None
            assert shape_fullsky_car is not None

            assert type(mask) == enmap.ndmap, 'The provided mask must be of type ndmap if the intensity map is as well provided with this type'
            box_coordinates = mask.reshape(shape_car).corners()

            take_box_function = lambda x: x.reshape(shape_fullsky_car).submap(box_coordinates).ravel()#[mask_bool]

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
        spherical_derivatives = get_second_spherical_derivatives_from_spin_derivatives(
            input_map=intensity_CMB,
            spin_derivatives_dict=intensity_spin_2_derivatives,
            shape_pixels_output=(intensity_CMB.size,),
            lmax=lmax,
            input_spin=input_spin
        )
    else:
        print("Using provided spherical derivatives", flush=True)
        assert 'phi_phi' in spherical_derivatives, "The provided spherical_derivatives dictionnary must contain the 'phi_phi' key for the second derivative with respect to phi (including factor 1/sin^2(theta))"
        assert 'theta_phi' in spherical_derivatives, "The provided spherical_derivatives dictionnary must contain the 'theta_phi' key for the second derivative with respect to theta and phi"
        assert 'theta_theta' in spherical_derivatives, "The provided spherical_derivatives dictionnary must contain the 'theta_theta' key for the second derivative with respect to theta"

        for spin in spherical_derivatives:
            if spherical_derivatives[spin].shape != np.prod(shape_car):
                spherical_derivatives[spin] = take_box_function(spherical_derivatives[spin])
        

    d2phi_minus_d2theta = spherical_derivatives['phi_phi'][mask_bool] - spherical_derivatives['theta_theta'][mask_bool]
    crossed_derivatives_imaginary_part = 2 * 1j * spherical_derivatives['theta_phi'][mask_bool]
    
    derivatives_maps = {f'{i}{j}':Spin_maps() for i in ['x','y'] for j in ['x','y']}

    derivatives_maps['xx'][-2] = (d2phi_minus_d2theta + crossed_derivatives_imaginary_part)/4
    derivatives_maps['xx'][0] = 0.5 * (spherical_derivatives['phi_phi'][mask_bool] + spherical_derivatives['theta_theta'][mask_bool])

    derivatives_maps['yy'][-2] = (-d2phi_minus_d2theta - crossed_derivatives_imaginary_part)/4
    derivatives_maps['yy'][0] = 0.5 * (spherical_derivatives['phi_phi'][mask_bool] + spherical_derivatives['theta_theta'][mask_bool])

    derivatives_maps['xy'][-2] = (-d2phi_minus_d2theta - crossed_derivatives_imaginary_part)/(4 * 1j)
    derivatives_maps['yx'][-2] = derivatives_maps['xy'][-2]

    for key in derivatives_maps:
        derivatives_maps[key][2] = np.conj(derivatives_maps[key][-2])
        # print("####", derivatives_maps[key][2].shape)

    differential_ellipticity_spin_maps = Spin_maps()

    for spin in [-2, 0, 2]:
        differential_ellipticity_spin_maps[spin] = np.zeros((delta_sigma.size, npix), dtype=np.complex128)
        for idx_0, key_0 in enumerate(['x', 'y']):
            for idx_1, key_1 in enumerate(['x', 'y']):
                if spin not in derivatives_maps[key_0+key_1]:
                    continue
                differential_ellipticity_spin_maps[spin] += contract(
                    'd,p->dp', 
                    alpha_2[:,idx_0,idx_1], 
                    derivatives_maps[key_0+key_1][spin].squeeze(), 
                    memory_limit='max_input'
                )
    
    if 0 not in differential_ellipticity_spin_maps:
        differential_ellipticity_spin_maps[0] = np.zeros_like(differential_ellipticity_spin_maps[2])

    if type(intensity_CMB) == enmap.ndmap:
        for spin in differential_ellipticity_spin_maps.spins:
            differential_ellipticity_spin_maps[spin] = enmap.ndmap(
                differential_ellipticity_spin_maps[spin],
                wcs=intensity_CMB.wcs
            )
    else:
        for spin in differential_ellipticity_spin_maps.spins:
            differential_ellipticity_spin_maps[spin] = differential_ellipticity_spin_maps[spin].squeeze()

    return differential_ellipticity_spin_maps
