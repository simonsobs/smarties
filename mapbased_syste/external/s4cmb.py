# S4CMB
# Copyright (c) 2016-2021 Julien Peloton, Giulio Fabbian.
#
# This file is part of s4cmb
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
Routines taken from CMBS4 and readapted including diverse tools to manipulate alms and maps from the s4cmb package.
"""
import numpy as np
import healpy as hp

def get_alpha_raise(s, lmax):
    """Response coefficient of spin-s spherical harmonic to spin raising operator.

    Author: Julien Carron (j.carron@sussex.ac.uk)

    Parameters
    ----------
    s : int
        Input spin of the spherical harmonic.
    lmax : int
        Maximum multipole moment.
    
    Returns
    -------
    ret : np.ndarray
        Response coefficient of spin-s spherical harmonic to spin raising operator.

    Notes
    -----
    The response coefficient is defined as:
        alpha(s, l) = sqrt((l - s) * (l + s + 1))
    where l is the multipole moment.
    The response coefficient is zero for l < |s|.
    """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s):] = np.sqrt(
        np.arange(abs(s) - s, lmax - s + 1) * np.arange(abs(s) + s + 1, lmax + s + 2)
    )
    return ret


def get_alpha_lower(s, lmax):
    """Response coefficient of spin-s spherical harmonic to spin lowering operator.

    Author: Julien Carron (j.carron@sussex.ac.uk)

    Parameters
    ----------
    s : int
        Input spin of the spherical harmonic.
    lmax : int
        Maximum multipole moment.

    Returns
    -------
    ret : np.ndarray
        Response coefficient of spin-s spherical harmonic to spin lowering operator.

    Notes
    -----
    The response coefficient is defined as:
        alpha(s, l) = sqrt((l + s) * (l - s + 1))
    where l is the multipole moment.
    The response coefficient is zero for l < |s|.
    """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(s):] = -np.sqrt(
        np.arange(s + abs(s), lmax + s + 1) * np.arange(abs(s) - s + 1, lmax - s + 2)
    )
    return ret

def get_first_spin_derivative(grad_curl_alms, nside, input_spin):
    """
    Function to obtain the maps after applying the spin-raising and spin-lowering operators on the input alms of arbitrary spin.

    Parameters
    ----------
    grad_curl_alms : list of np.ndarray
        List of two arrays containing the grad and curl parts of the spherical harmonic coefficients.
        grad_curl_alms[0] is the grad part and grad_curl_alms[1] is the curl part, and the last dimension is the healpix ordering
        of the alms.
    nside : int
        Healpix nside parameter.
    input_spin : int
        Input spin of the spherical harmonic coefficients grad_curl_alms.
    
    Returns
    -------
    dictionary_spin_derivative : dict
        Dictionary containing the spin-s transform of the input spherical harmonic with keys being:
        * '1': map after application of the spin-raising operator on the input alms
        * '-1': map after application of the spin-lowering operator on the input alms

    """

    assert input_spin >= 0, input_spin
    assert hp.Alm.getlmax(grad_curl_alms[0].size) == hp.Alm.getlmax(grad_curl_alms[1].size)
    lmax = hp.Alm.getlmax(grad_curl_alms[0].size)
    # shape (2, 12 * nside ** 2),

    # First obtaining the application of the spin-lowering operator on the input alms
    _gclm = [
        hp.almxfl(grad_curl_alms[0], get_alpha_raise(input_spin, lmax)),
        hp.almxfl(grad_curl_alms[1], get_alpha_raise(input_spin, lmax)),
    ]
    spin_raised_maps = np.array(hp.alm2map_spin(_gclm, nside, input_spin+1, lmax))

    # Second obtaining the application of the spin-raising operator on the input alms
    if input_spin == 0:
        spin_lowered_maps = np.copy(spin_raised_maps)
        spin_lowered_maps[1] *= -1
    else:
        _gclm = [
            hp.almxfl(grad_curl_alms[0], get_alpha_lower(input_spin, lmax)),
            hp.almxfl(grad_curl_alms[1], get_alpha_lower(input_spin, lmax)),
            ]
        spin_lowered_maps = np.array(hp.alm2map_spin(_gclm, nside, input_spin-1, lmax))

    
    return {
        input_spin+1: spin_raised_maps[0] + 1j * spin_raised_maps[1], 
        input_spin-1: spin_lowered_maps[0] + 1j * spin_lowered_maps[1], 
    } 


def get_second_spin_derivative(grad_curl_alms, nside, input_spin):
    """
    Function to obtain the maps after applying any combination of two applications of the spin-raising and spin-lowering operators on the input alms of arbitrary spin. 

    Parameters
    ----------
    grad_curl_alms : list of np.ndarray
        List of two arrays containing the grad and curl parts of the spherical harmonic coefficients.
        grad_curl_alms[0] is the grad part and grad_curl_alms[1] is the curl part, and the last dimension is the healpix ordering
        of the alms.
    nside : int
        Healpix nside parameter.
    input_spin : int
        Input spin of the spherical harmonic coefficients grad_curl_alms.
    
    Returns
    -------
    dictionary_spin_derivative : dict
        Dictionary containing the spin-s transform of the input spherical harmonic with keys being:
        * '2': map after application of two spin-raising operator on the input alms
        * '-2': map after application of two spin-lowering operator on the input alms
        * '+1-1': map after application of the spin-raising then the spin-lowering operators on the input alms
        * '-1+1': map after application of the spin-lowering then the spin-raising operators on the input alms

    """

    assert input_spin >= 0, input_spin
    assert hp.Alm.getlmax(grad_curl_alms[0].size) == hp.Alm.getlmax(grad_curl_alms[1].size)
    lmax = hp.Alm.getlmax(grad_curl_alms[0].size)
    # shape (2, 12 * nside ** 2),

    assert input_spin >= 0, input_spin
    assert hp.Alm.getlmax(grad_curl_alms[0].size) == hp.Alm.getlmax(grad_curl_alms[1].size)
    lmax = hp.Alm.getlmax(grad_curl_alms[0].size)
    # shape (2, 12 * nside ** 2),

    # First obtaining the application of two successsive spin-raising operators on the input alms
    _gclm = [
        hp.almxfl(alms, get_alpha_lower(input_spin, lmax)*get_alpha_lower(input_spin-1, lmax)) for alms in grad_curl_alms
    ]
    if input_spin <= 1:
        spin_2_lowered_maps = -np.array([hp.alm2map(alms, nside) for alms in _gclm])
    else:
        spin_2_lowered_maps = np.array(hp.alm2map_spin(_gclm, nside, input_spin-2, lmax))

    # Second obtaining the application of two successsive spin-lowering operators on the input alms
    _gclm = [
        hp.almxfl(alms, get_alpha_raise(input_spin, lmax)*get_alpha_raise(input_spin+1, lmax)) for alms in grad_curl_alms
    ]
    spin_2_raised_maps = np.array(hp.alm2map_spin(_gclm, nside, input_spin+2, lmax))

    # Third obtaining the application of the spin-raising then the spin-lowering operators on the input alms
    _gclm = [
        hp.almxfl(alms, get_alpha_raise(input_spin, lmax)*get_alpha_lower(input_spin+1, lmax)) for alms in grad_curl_alms
    ]
    if input_spin == 0:
        spin_raised_lowered_maps = -np.array([hp.alm2map(alms, nside) for alms in _gclm])
    else:
        spin_raised_lowered_maps = np.array(hp.alm2map_spin(_gclm, nside, input_spin, lmax))

    # Fourth obtaining the application of the spin-lowering then the spin-raising operators on the input alms
    _gclm = [
        hp.almxfl(alms, get_alpha_lower(input_spin, lmax)*get_alpha_raise(input_spin-1, lmax)) for alms in grad_curl_alms
    ]
    if input_spin == 0:
        spin_lowered_raised_maps = -np.array([hp.alm2map(alms, nside) for alms in _gclm])
    else:
        spin_lowered_raised_maps = np.array(hp.alm2map_spin(_gclm, nside, input_spin, lmax))

    return {
        input_spin+2: spin_2_raised_maps[0] + 1j * spin_2_raised_maps[1], 
        input_spin-2: spin_2_lowered_maps[0] + 1j * spin_2_lowered_maps[1], 
        '+1-1': spin_lowered_raised_maps[0] + 1j * spin_lowered_raised_maps[1],
        '-1+1': spin_raised_lowered_maps[0] + 1j * spin_raised_lowered_maps[1],
    } 
