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
import h5py
from opt_einsum import contract
from pixell import enmap

from smarties.hn import Spin_maps, Spin_nm

list_conventions = [
    'Third flattening', 
    'Third eccentricity',
    'Modified second flattening',
    'Plus-Cross ellipticity'
]

def get_coupled_spin(reference_spin, available_h_n_spin, available_signal_spins):
    """
    Get the coupled spins for a reference spin $k$ given a set of $h_n$ and signal spin maps, involved in a typical sum: 
        $ sum_{k' = -\infty}^{\infty} h_{k-k'} S_{k'}$
        
    Parameters
    ----------
    reference_spin: int
        Reference spin $k$
    available_h_n_spin: list[int]
        List of available $h_n$ spins, typically [-4, -2, 2, 4]
    available_signal_spins: list[int]
        List of available signal spins, typically [-2, 2]
    
    Returns
    -------
    coupled_spin: list[tuple]
        List of available coupled spins, each tuple being the coupled spins $(k-k', k')$
    """

    minimum_spin = np.min(list(available_h_n_spin) + list(available_signal_spins))
    maximum_spin = np.max(list(available_h_n_spin) + list(available_signal_spins))

    coupled_spin = []
    for spin in range(minimum_spin, maximum_spin+1):
        if reference_spin - spin in available_h_n_spin and spin in available_signal_spins:
            coupled_spin.append((reference_spin - spin, spin))
    return coupled_spin


def get_row_mapmaking_matrix(
        reference_spin, 
        h_n_spin_dict, 
        list_spin_input,
        polar_angle_coeff=None,
        dtype=complex
    ):
    """
    The mapmaking matrix will always be multiplied to the vector ordered with spins [0, 2, -2] for the $\tilde{S}^{\rm pixel}_{k}$ term.

    The $h_n$ dictionary must have the following structure:
        h_n_spin_dict = {spin: np.array([n_det, n_pix])} (for spin != 0)
        h_n_spin_dict = {0: np.array([1, 1])} (for spin = 0)
    And the $h_n$ maps will be summed over the detectors in the mapmaking matrix, so that the mapmaking matrix will be of shape [n_spin, n_pix] with n_spin the number of spins involved in list_spin_input and n_pix the number of pixels in the $h_n$ maps.

    Parameters
    ----------
    reference_spin: int
        Reference spin $k$ for the mapmaking matrix, typically 0, 2 or -2
    h_n_spin_dict: Spin_maps
        Dictionary of the summed $h_n$ maps, with the keys being the spins and the values the $h_n$ maps
    list_spin_input: list[int]
        List of spins involved in the input signal maps, typically [-2, 2] for polarization maps
    
    Returns
    -------
    mapmaking_matrix_row: np.ndarray
        Row of the mapmaking matrix of shape [n_pix, n_spin] with n_spin the number of spins involved in list_spin_input and n_pix the number of pixels in the $h_n$ maps.
        The row is given by list_spin_input.
    """
    if polar_angle_coeff is None:
        polar_angle_coeff = {spin:np.ones(h_n_spin_dict[0].shape[0]) for spin in h_n_spin_dict.spins}

    factor_func = lambda x: 1 if x == 0 else .5

    mapmaking_matrix_row = np.zeros(
        tuple(h_n_spin_dict[2].shape[1:]) + ( 
         len(list_spin_input),
        ), 
        dtype=dtype
    )
    for i, spin_name in enumerate(list_spin_input):
        mapmaking_matrix_row[:,i] = (
            factor_func(reference_spin) 
            * factor_func(spin_name) 
            * contract(
                'd...,d...->...', 
                polar_angle_coeff[spin_name-reference_spin], 
                h_n_spin_dict[spin_name-reference_spin])
            )

    return mapmaking_matrix_row

def get_rotation_matrix(angle):
    """
    Get the rotation matrix for a given angle.
    
    Parameters
    ----------
    angle: np.ndarray
        Angle in radians

    Returns
    -------
    rotation_matrix: np.ndarray
        Rotation matrix of shape (angle.shape, 2, 2), with the first dimension being the same as the input angle
    """

    angle = np.asarray(angle)

    rotation_matrix = np.zeros(angle.shape + (2, 2))
    rotation_matrix[...,0,0] = np.cos(angle)
    rotation_matrix[...,0,1] = np.sin(angle)
    rotation_matrix[...,1,0] = -np.sin(angle)
    rotation_matrix[...,1,1] = np.cos(angle)

    return rotation_matrix

def transform_array_maps_into_spin_maps(
        array_maps, 
        n_stokes_output=None
    ):
    """
    Transform an array of maps into a Spin_maps object,
    inheriting from the dictionary structure as
      {key:element} being the spin and the corresponding map, respectively.
    The transformation is done as follows:
        * The spin 0 field is assumed to be the first Stokes parameter (temperature) if n_stokes = 1 or 3.
        * The spin -2 field is assumed to be given by $0.5 * (Q - iU)$
        * The spin 2 field is assumed to be given by $0.5 * (Q + iU)$
    where Q and U are the second and third Stokes parameters, respectively, if n_stokes = 2 or 3.
      
    Parameters
    ----------
    array_maps: np.ndarray | enmap.ndmap
        Array of maps of shape (..., n_stokes, n_pix) with
        * if n_stokes = 1, the temperature field [T] is assumed to be provided (spin=0)
        * if n_stokes = 2, the polarization field [Q,U] is assumed to be provided (spin=2, -2)
        * if n_stokes = 3, the full Stokes parameters [T,Q,U] are assumed to be provided (spin=0, 2, -2)

    Returns
    -------
    spin_maps: Spin_maps
        Spin_maps object with keys being the spins and values being the corresponding maps

    """
    
    # if type(array_maps) == np.ndarray:
    n_stokes = array_maps.shape[-2] if array_maps.ndim > 1 else 1
    dimension_stokes = -2
    # elif type(array_maps) == enmap.ndmap:
    #     n_stokes = array_maps.shape[-3] if array_maps.ndim > 2 else 1
    #     dimension_stokes = -3

    assert n_stokes in [1, 2, 3], 'The number of Stokes parameters must be 1 (only temperature), 2 (only polarization) or 3 (both temperature and polarization)'

    if n_stokes_output is not None:
        assert n_stokes_output in [1, 2, 3], 'The number of Stokes parameters must be 1 (only temperature), 2 (only polarization) or 3 (both temperature and polarization)'
        assert not((n_stokes_output == 1) and (n_stokes == 2)), 'Incompatible Stokes parameter configurations'
    else:
        n_stokes_output = n_stokes

    output_spin_maps = Spin_maps()
    
    
    if n_stokes == 1 or n_stokes == 3:
        
        if array_maps.ndim == 1:
            # Only temperature field is provided
            index_T = (...,)
        else:
            index_T = tuple(array_maps.shape[:dimension_stokes]) + (0,)
        output_spin_maps[0] = array_maps[*index_T] # [spin=0]

    if n_stokes >= 2:
        output_spin_maps[-2] = .5*(
            array_maps[...,dimension_stokes,:] - 1j * array_maps[...,dimension_stokes+1,:]
        ) # [spin=-2]
        output_spin_maps[2] = .5*(
            array_maps[...,dimension_stokes,:] + 1j * array_maps[...,dimension_stokes+1,:]
        ) # [spin=2]
    
    if n_stokes_output != n_stokes:
        if n_stokes == 1:
            output_spin_maps[2] = np.zeros_like(output_spin_maps[0], dtype=complex)
            output_spin_maps[-2] = np.zeros_like(output_spin_maps[0], dtype=complex)
        elif n_stokes == 2:
            output_spin_maps[0] = np.zeros_like(output_spin_maps[-2], dtype=complex)
    
    if type(array_maps) == enmap.ndmap:
        for spin in output_spin_maps.spins:
            output_spin_maps[spin] = enmap.ndmap(
                output_spin_maps[spin], 
                wcs=array_maps.wcs
            )

    return output_spin_maps
    
def transform_spin_maps_into_array_maps(
        spin_maps
    ):
    """
    Transform a Spin_maps object into an array of maps, 
    as:
        * the spin 0 field is assumed to be the first Stokes parameter (temperature).
        * the spin -2 field is assumed to be given by $0.5 * (Q - iU)$
        * the spin 2 field is assumed to be given by $0.5 * (Q + iU)$
    where Q and U are the second and third Stokes parameters, respectively, if n_stokes = 2 or 3. 
    
    Parameters
    ----------
    spin_maps: Spin_maps
        Spin_maps object to transform each with keys being the spins and values being the corresponding maps
        associated to the dimension (..., n_pix) where n_pix is the number of pixels in the maps.

    Returns
    -------
    array_maps: np.ndarray
        Array of maps of shape (..., n_stokes, n_pix)
    """
    
    n_stokes = 0
    if 0 in spin_maps:
        n_stokes += 1
        first_spin = 0
    if -2 in spin_maps and 2 in spin_maps:
        n_stokes += 2
        first_spin = -2
    assert n_stokes in [1, 2, 3], 'The number of Stokes parameters must be 1 (only temperature), 2 (only polarization) or 3 (both temperature and polarization)'
    boolean_car_pixelization = type(spin_maps[first_spin]) == enmap.ndmap
    dimension_pixels = -1 if not boolean_car_pixelization else -2
    
    shape_pix = (spin_maps[first_spin].shape[-dimension_pixels:],) 
    dtype = spin_maps[first_spin].dtype

    array_maps = np.zeros(spin_maps[first_spin].shape[:-dimension_pixels] + (n_stokes,) + shape_pix, dtype=dtype)
    
    if n_stokes == 1 or n_stokes == 3:
        # Only temperature field is provided
        array_maps[...,0,:] = spin_maps[0]
    if n_stokes >= 2:
        array_maps[...,1,:] = spin_maps[-2] + spin_maps[2]  # [Q, U] -> spin -2 and 2
        array_maps[...,2,:] = -1j * (spin_maps[2] - spin_maps[-2])
    
    if boolean_car_pixelization:
        array_maps = enmap.ndmap(array_maps, wcs=spin_maps[first_spin].wcs)
    return array_maps


def save_partial_spin_maps(
        partial_spin_maps, 
        nstokes,
        shape_pixels,
        mask_on_full_map, 
        path_output,
        format_output='.npy'
    ):

    extended_final_maps = np.zeros((nstokes,)+(np.prod(shape_pixels),), dtype=complex)
    if nstokes == 3 or nstokes == 1:
        extended_final_maps[0, mask_on_full_map != 0] = partial_spin_maps[0]
    if nstokes == 3 or nstokes == 2:
        final_Q_map = (partial_spin_maps[-2] + partial_spin_maps[2])/2.
        final_U_map = 1j*(partial_spin_maps[-2] - partial_spin_maps[2])/2.

        extended_final_maps[-2, mask_on_full_map != 0] = final_Q_map.real
        extended_final_maps[-1, mask_on_full_map != 0] = final_U_map.real

    if extended_final_maps.shape[-len(shape_pixels):] != shape_pixels:
        extended_final_maps = extended_final_maps.reshape(
            extended_final_maps.shape[:-1] + shape_pixels
        )
    
    is_car = False
    first_spin = 0 if (nstokes == 1 or nstokes == 3) else -2
    if type(partial_spin_maps[first_spin]) == enmap.ndmap:
        extended_final_maps = enmap.ndmap(
            extended_final_maps, 
            wcs=partial_spin_maps[first_spin].wcs
        )
        is_car = True

    if is_car:
        if format_output == '.fits' and not path_output.endswith('.fits'):
            path_output = path_output + '.fits'
        elif format_output == '.hdf' and not path_output.endswith('.hdf'):
            path_output = path_output + '.hdf'
        enmap.write_map(
                path_output, 
                extended_final_maps,
                extra={'BUNIT' : 'uK'}
            )
    elif format_output == '.npy' or path_output.endswith('.npy'):
        if not path_output.endswith('.npy'):
            path_output = path_output + '.npy'
        print("Saving map into", path_output)
        np.save(path_output, extended_final_maps[:,mask_on_full_map!=0])
    elif format_output == '.fits' or path_output.endswith('.fits'):
        if not path_output.endswith('.fits'):
            path_output = path_output + '.fits'
        print("Saving map into", path_output)
        
        hp.write_map(
            path_output, 
            extended_final_maps, 
            overwrite=True
        )
            
    elif format_output in ['.hdf', '.hdf5'] or path_output.endswith('.hdf') or path_output.endswith('.hdf5'):
        if not (path_output.endswith('.hdf') or path_output.endswith('.hdf5')):
            path_output = path_output + '.hdf'
        print("Saving map into", path_output)
    
        with h5py.File(path_output, 'w') as hf:
            hf.create_dataset('maps', data=extended_final_maps)
        hf.close()
    else:
        raise ValueError("Unsupported format_output. Supported formats are '.npy' and '.fits'.")


def convert_ellipticities_conventions(
        dictionary_ellipticities, 
        sigma_FWHM,  # arcmin
        input_ellipticity_convention='Third flattening',
        output_ellipticity_convention='Third flattening'
    ):
    """
    Convert ellipticity parameters from one convention to another.
    The current supported conventions are:
        * 'Third flattening': f = (a-b)/(a+b) = (sigma_maj - sigma_min)/(sigma_maj + sigma_min)
        * 'Third eccentricity': e = sqrt((a^2 - b^2)/a^2) = sqrt(sigma_maj^2 - sigma_min^2)/(sigma_maj^2 + sigma_min^2)
        * 'Modified second flattening': e = a/b = sigma_maj / sigma_min
        * 'Plus-Cross ellipticity': dp = 2 f cos (2 theta), dc = 2 f sin (2 theta), with f the third flattening and theta the ellipticity angle.


    Parameters
    ----------
    dictionary_ellipticities: dict
        Dictionary containing the ellipticity parameters to convert.
        The keys must be:
            * 'ellipticity_value' and 'ellipticity_angle' for 'Third flattening', 
            'Third eccentricity' and 'Modified second flattening' conventions.
            * 'dp' and 'dc' for 'Plus-Cross ellipticity' convention.
    sigma_FWHM: float or np.ndarray
        Beam full-width at half-maximum (FWHM) in arcminutes.
    input_ellipticity_convention: str
        Convention of the input ellipticity parameters. Must be one of the following:
            * 'Third flattening'
            * 'Third eccentricity'
            * 'Modified second flattening'
            * 'Plus-Cross ellipticity'
    output_ellipticity_convention: str
        Convention of the output ellipticity parameters. Must be one of the following:
            * 'Third flattening'
            * 'Third eccentricity'
            * 'Modified second flattening'
            * 'Plus-Cross ellipticity'
        
    Returns
    -------
    converted_ellipticities: dict
        Dictionary containing the converted ellipticity parameters in the desired convention.
    """
    assert input_ellipticity_convention in list_conventions, "ellipticity_parameter_convention must be an element of the list of supported conventions {list_conventions}"
    assert output_ellipticity_convention in list_conventions, "ellipticity_parameter_convention must be an element of the list of supported conventions {list_conventions}"

    sigma_cs = np.asarray(sigma_FWHM) / ((8 * np.log(2)) ** 0.5) * np.pi/(180*60)

    for key, item in dictionary_ellipticities.items():
        dictionary_ellipticities[key] = np.asarray(item)

    # if input_ellipticity_convention == output_ellipticity_convention:
    #     return dictionary_ellipticities

    if input_ellipticity_convention == 'Plus-Cross ellipticity':
        delta_sigma = np.sqrt(
                dictionary_ellipticities['dc']**2 + dictionary_ellipticities['dp']**2
            ) * sigma_cs / 2. 

        ellipticity_angle = np.arctan2(
            dictionary_ellipticities['dc'], 
            dictionary_ellipticities['dp']
        ) /2.

    elif input_ellipticity_convention == 'Modified second flattening':
        # Modified second flattening
        # e = a/b = sigma_maj / sigma_min

        assert np.all(dictionary_ellipticities['ellipticity_value'] >= 1), "For the Modified second flattening convention, ellipticity_value must be >= 1, with value 1 corresponding to a circular beam."

        delta_sigma = sigma_cs * (dictionary_ellipticities['ellipticity_value'] - 1) / (dictionary_ellipticities['ellipticity_value'] + 1)

        ellipticity_angle = dictionary_ellipticities['ellipticity_angle']

    elif input_ellipticity_convention == 'Third flattening':
        # Third flattening
        # f = (a-b)/(a+b) = (sigma_maj - sigma_min)/(sigma_maj + sigma_min)

        assert np.all(
            np.logical_and(
                dictionary_ellipticities['ellipticity_value'] >= 0, 
                dictionary_ellipticities['ellipticity_value'] <= 1
            )
        ), "For the Third flattening convention, ellipticity_value must be between 0 and 1, with value 0 corresponding to a circular beam."
        
        delta_sigma = dictionary_ellipticities['ellipticity_value'] * sigma_cs
        ellipticity_angle = dictionary_ellipticities['ellipticity_angle']
        
    elif input_ellipticity_convention == 'Third eccentricity':
        # Third eccentricity
        # e = sqrt((a^2 - b^2)/a^2) = sqrt(sigma_maj^2 - sigma_min^2)/(sigma_maj^2 + sigma_min^2)j

        assert np.all(
            np.logical_and(
                dictionary_ellipticities['ellipticity_value'] >= 0, 
                dictionary_ellipticities['ellipticity_value'] < 1
            )
        ), "For the Third eccentricity convention, ellipticity_value must be between 0 and 1, with value 0 corresponding to a circular beam."

        delta_sigma = np.where(
            dictionary_ellipticities['ellipticity_value'] != 0,
            2 * sigma_cs * (1 - np.sqrt(1 - dictionary_ellipticities['ellipticity_value'] ** 2)) / dictionary_ellipticities['ellipticity_value'],
            0
        )/2. # the formula is not defined for ellipticity = 0, which correspond to a circular beam where the deviation is 0
        ellipticity_angle = dictionary_ellipticities['ellipticity_angle']
    else:
        raise ValueError("input_ellipticity_convention must be an element of the list of supported conventions {list_conventions}")




    if output_ellipticity_convention == 'Plus-Cross ellipticity':
        ellipticity_value_dp = (delta_sigma * 2. / sigma_cs) * np.cos(2 * ellipticity_angle)
        ellipticity_value_dc = (delta_sigma * 2. / sigma_cs) * np.sin(2 * ellipticity_angle)
        
        return {
            'dc': ellipticity_value_dc,
            'dp': ellipticity_value_dp,
            'ellipticity_parameter_convention': 'Plus-Cross ellipticity'
        }
    elif output_ellipticity_convention == 'Modified second flattening':
        ellipticity_value = (sigma_cs + delta_sigma) / (sigma_cs - delta_sigma)
        return {
            'ellipticity_value': ellipticity_value,
            'ellipticity_angle': ellipticity_angle,
            'ellipticity_parameter_convention': 'Modified second flattening'
        }
    elif output_ellipticity_convention == 'Third flattening':
        ellipticity_value = delta_sigma / sigma_cs
        return {
            'ellipticity_value': ellipticity_value,
            'ellipticity_angle': ellipticity_angle,
            'ellipticity_parameter_convention': 'Third flattening'
        }
    elif output_ellipticity_convention == 'Third eccentricity':
        ellipticity_value = (
            (sigma_cs + delta_sigma)**2 - (sigma_cs - delta_sigma)**2
            ) / (
                (sigma_cs + delta_sigma)**2 + (sigma_cs - delta_sigma)**2
            )
        return {
            'ellipticity_value': ellipticity_value,
            'ellipticity_angle': ellipticity_angle,
            'ellipticity_parameter_convention': 'Third eccentricity'
        }
    else:
        raise ValueError("output_ellipticity_convention must be an element of the list of supported conventions {list_conventions}")
    

def flatten_CAR_maps(maps_CAR):
    first_dimensions = maps_CAR.shape[:-2] if maps_CAR.shape[:-2] != (1,) else tuple()
    return maps_CAR.reshape(first_dimensions + (np.prod(maps_CAR.shape[-2:]),))

def unflatten_CAR_maps(maps_CAR_flatten, original_shape_pixels):
    return maps_CAR_flatten.reshape(maps_CAR_flatten.shape[:-1] + original_shape_pixels)

def reweight_h_maps(
        h_dictionary: dict| Spin_maps,
        list_weights: np.ndarray,
        new_weighting_bool: bool,
        error_precision: float,
        list_spin: list[int],
):
    if 0 not in h_dictionary.spins and Spin_nm((0,0)) not in h_dictionary.spins:
        spin_0 = Spin_nm((0,0)) if len(list(h_dictionary.spins)[0]) == 2 else 0

        h_dictionary[spin_0] = list_weights[..., np.newaxis] if list_weights.ndim == 1 else list_weights
        if new_weighting_bool:
            print("--Applying new weighting to h_n dictionary...", flush=True)

            array_hits_detector_pixel = np.int64(np.logical_or(
                np.abs(h_dictionary[list_spin[0]]) > 10*error_precision,
                np.abs(h_dictionary[-list_spin[0]]) > 10*error_precision
                )
            )
            pixel_map_weighting_array = (array_hits_detector_pixel).sum(axis=0)
            # pixel_map_weighting_array = np.sum(new_weighting_array, axis=0)

            if 0 in pixel_map_weighting_array:
                print("--Warning: some pixels are not observed by any detector, they will have their h maps values set to zero", flush=True)
                print("--Number of unobserved pixels: {}".format(np.sum(pixel_map_weighting_array==0)), flush=True)
                print("--Total number of pixels: {}".format(pixel_map_weighting_array.size), flush=True)
                print("--Fraction of unobserved pixels: {:.2e}".format(np.sum(pixel_map_weighting_array==0)/pixel_map_weighting_array.size), flush=True)
            
            for spin in h_dictionary.spins:
                if spin != spin_0:
                    h_dictionary[spin][...,pixel_map_weighting_array!=0] = contract(
                        'd...,d...,...->d...',
                        h_dictionary[spin], 
                        list_weights,
                        1 / pixel_map_weighting_array[pixel_map_weighting_array!=0]
                    )
                
            h_dictionary[spin_0] = np.where(array_hits_detector_pixel==0, 0, array_hits_detector_pixel / pixel_map_weighting_array )
    else:
        spin_0 = 0 if 0 in h_dictionary.spins else Spin_nm((0,0))
        print("--Spin 0 in h maps dictionary, re-weighting applied.", flush=True)
        cond_non_zero = h_dictionary[spin_0] != 0
        print(h_dictionary[spin_0].shape, flush=True)
        inverse_weights = 1 / list_weights[...,None] if list_weights.ndim == 1 else np.where(cond_non_zero, 1 / list_weights[...,cond_non_zero], 0)
        
        sum_hits = h_dictionary[spin_0].sum(axis=0)

        for spin in h_dictionary.spins:
            if spin != spin_0:
                h_dictionary[spin][cond_non_zero] *= (inverse_weights * h_dictionary[spin_0] / sum_hits)[cond_non_zero]
                    
        
        h_dictionary[spin_0] = np.where(cond_non_zero, h_dictionary[spin_0] / sum_hits, 0)
    return h_dictionary
