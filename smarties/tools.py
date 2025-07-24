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

from smarties.hn import Spin_maps

def get_coupled_spin(reference_spin, available_h_n_spin, available_signal_spins):
    """
    Get the coupled spins for a reference spin $k$ given a set of $h_n$ and signal spin maps, involved in a typical sum: 
        $ \sum_{k' = -\infty}^{\infty} h_{k-k'} S_{k'}$
        
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


def get_row_mapmaking_matrix(reference_spin, h_n_spin_dict, list_spin_input):
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

    factor_func = lambda x: 1 if x == 0 else .5

    mapmaking_matrix_row = np.zeros((h_n_spin_dict[2].shape[-1], len(list_spin_input)), dtype=complex)
    for i, spin_name in enumerate(list_spin_input):
        mapmaking_matrix_row[:,i] = factor_func(reference_spin) * factor_func(spin_name) * h_n_spin_dict[spin_name-reference_spin].sum(axis=0)
        
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
    rotation_matrix[...,0,0] = np.cos(2 * angle)
    rotation_matrix[...,0,1] = -np.sin(2 * angle)
    rotation_matrix[...,1,0] = np.sin(2 * angle)
    rotation_matrix[...,1,1] = np.cos(2 * angle)

    return rotation_matrix

def transform_array_maps_into_spin_maps(array_maps):
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
    array_maps: np.ndarray
        Array of maps of shape (..., n_stokes, n_pix) with
        * if n_stokes = 1, the temperature field [T] is assumed to be provided (spin=0)
        * if n_stokes = 2, the polarization field [Q,U] is assumed to be provided (spin=2, -2)
        * if n_stokes = 3, the full Stokes parameters [T,Q,U] are assumed to be provided (spin=0, 2, -2)

    Returns
    -------
    spin_maps: Spin_maps
        Spin_maps object with keys being the spins and values being the corresponding maps

    """
    
    n_stokes = array_maps.shape[-2] if array_maps.ndim > 1 else 1
    assert n_stokes in [1, 2, 3], 'The number of Stokes parameters must be 1 (only temperature), 2 (only polarization) or 3 (both temperature and polarization)'

    output_spin_maps = Spin_maps()
    
    if n_stokes == 1 or n_stokes == 3:
        # Only temperature field is provided
        output_spin_maps[0] = array_maps[...,0,:] # [spin=0]
    
    if n_stokes >= 2:
        output_spin_maps[-2] = .5*(array_maps[...,-2,:] - 1j * array_maps[...,-1,:]) # [spin=-2]
        output_spin_maps[2] = .5*(array_maps[...,-2,:] + 1j * array_maps[...,-1,:]) # [spin=2]
    
    return output_spin_maps
    
def transform_spin_maps_into_array_maps(spin_maps):
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
    if -2 in spin_maps and 2 in spin_maps:
        n_stokes += 2
    assert n_stokes in [1, 2, 3], 'The number of Stokes parameters must be 1 (only temperature), 2 (only polarization) or 3 (both temperature and polarization)'
    n_pix = spin_maps[0].shape[-1] if n_stokes == 1 else spin_maps[-2].shape[-1]
    dtype = spin_maps[0].dtype if n_stokes == 1 else spin_maps[-2].dtype

    array_maps = np.zeros(spin_maps[0].shape[:-1] + (n_stokes, n_pix), dtype=dtype)
    
    if n_stokes == 1 or n_stokes == 3:
        # Only temperature field is provided
        array_maps[...,0,:] = spin_maps[0]
    if n_stokes >= 2:
        array_maps[...,1,:] = spin_maps[-2] + spin_maps[2]  # [Q, U] -> spin -2 and 2
        array_maps[...,2,:] = -1j * (spin_maps[2] - spin_maps[-2])
        
    return array_maps
