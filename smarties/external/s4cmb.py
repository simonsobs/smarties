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
from os import cpu_count
import numpy as np
import healpy as hp
import ducc0



def _ducc_kwargs(
        spin, 
        nside, 
        lmax, 
        mmax=None):
    
    ducc_healpix_obj = ducc0.healpix.Healpix_Base(nside, 'RING')
    if mmax is None:
        mmax = lmax
    m_array = np.arange(mmax + 1)
    kwargs = {'spin': spin,
              'lmax': lmax, 
              'mmax': mmax,
              'mstart': (m_array*(2*lmax+1-m_array)//2).astype(np.uint64, copy=False), 
              **ducc_healpix_obj.sht_info()
}
    return kwargs

def _alm2map_ducc0(alm, spin, nside, lmax=None, mmax=None, nthreads=-1):

    if nthreads < 0:
        nthreads = cpu_count()

    if alm.ndim > 1:
        alm_size = alm.shape[-1]
    else:
        alm_size = alm.size
    if lmax is None:
        lmax = hp.Alm.getlmax(alm.shape[-1])
    else:
        assert lmax <= hp.Alm.getlmax(alm.shape[-1]), (lmax, hp.Alm.getlmax(alm.shape[-1]))
    if mmax is None:
        mmax = lmax

    maps = ducc0.sht.synthesis(
        alm=np.atleast_2d(alm),
        nthreads=nthreads,
        **_ducc_kwargs(
            spin, 
            nside, 
            lmax, 
            mmax, 
        )
    )
    return maps


def _map2alm_ducc0(maps, spin, lmax=None, mmax=None, nthreads=-1):

    nside = hp.npix2nside(maps.shape[-1])
    
    if lmax is None:
        lmax = 3 * nside - 1

    if mmax is None:
        mmax = lmax

    if nthreads < 0:
        nthreads = cpu_count()

    weight = 4*np.pi/(12 * nside**2)
    alm = ducc0.sht.adjoint_synthesis(
        map=np.atleast_2d(maps) * weight, 
        nthreads=nthreads,
        **_ducc_kwargs(
            spin, 
            nside, 
            lmax, 
            mmax, 
        )
    )
    return alm

def map2alm_ducc0_iter(maps, spin, lmax=None, mmax=None, iter=3):
    nside = hp.npix2nside(maps.shape[-1])
    alms = _map2alm_ducc0(maps, spin=spin, lmax=lmax, mmax=mmax)

    for iter_ in range(iter):
        residual_map = _alm2map_ducc0(alms, spin=spin, nside=nside, lmax=lmax, mmax=mmax) - maps
        alms -= _map2alm_ducc0(residual_map, spin=spin, lmax=lmax, mmax=mmax)
    return alms

def get_healpix_ring_pixel_layout(nside, th_idx):
    """Healpix ring layout.

    From 'get_pixel_layout' subroutine in healpix f90 package.

    Author: Julien Carron (j.carron@sussex.ac.uk)

    Parameters
    ----------
    nside : int
        Healpix nside parameter.
    th_idx : int
        Ring index (0 <= th_idx < 4*nside - 1).

    Returns
    -------
    startpix : int
        Starting pixel number.
    nphi : int
        Number of pixels in the ring.
    kphi0 : int
        Starting pixel number in the ring.
    cth : float
        Cosine of the polar angle.
    sth : float
        Sine of the polar angle.
    """
    ith = th_idx + 1
    nrings = 2 * nside
    assert 1 <= ith <= 4 * nside - 1, (ith, nrings)
    if ith > nrings:
        startpix, nphi, kphi0, cth, sth = get_healpix_ring_pixel_layout(
            nside, ith - 2 * (ith - nrings) - 1
        )
        return 12 * nside ** 2 - startpix - nphi, nphi, kphi0, -cth, sth
    dth1 = 1.0 / 3.0 / nside ** 2
    dth2 = 2.0 / 3.0 / nside
    dst1 = 1.0 / (np.sqrt(6.0) * nside)
    if ith < nside:  # polar cap (north)
        cth = 1.0 - ith ** 2 * dth1
        nphi = 4 * ith
        kphi0 = 1
        sth = np.sin(2.0 * np.arcsin(ith * dst1))
        startpix = 2 * ith * (ith - 1)
    else:
        cth = (2 * nside - ith) * dth2
        nphi = 4 * nside
        kphi0 = (ith + 1 - nside) % 2
        sth = np.sqrt((1.0 - cth) * (1.0 + cth))
        startpix = 2 * nside * (nside - 1) + (ith - nside) * int(nphi)
    return startpix, nphi, kphi0, cth, sth

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
    _gclm = np.array([
        hp.almxfl(grad_curl_alms[0], get_alpha_raise(input_spin, lmax)),
        hp.almxfl(grad_curl_alms[1], get_alpha_raise(input_spin, lmax)),
    ])
    spin_raised_maps = np.array(_alm2map_ducc0(_gclm, nside=nside, spin=input_spin+1, lmax=lmax))

    # Second obtaining the application of the spin-raising operator on the input alms
    if input_spin == 0:
        spin_lowered_maps = np.copy(spin_raised_maps)
        spin_lowered_maps[1] *= -1
    else:
        _gclm = np.array([
            hp.almxfl(grad_curl_alms[0], get_alpha_lower(input_spin, lmax)),
            hp.almxfl(grad_curl_alms[1], get_alpha_lower(input_spin, lmax)),
            ])
        spin_lowered_maps = np.array(_alm2map_ducc0(_gclm, nside=nside, spin=input_spin-1, lmax=lmax))

    
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
    _gclm = np.array([
        hp.almxfl(alms, get_alpha_lower(input_spin, lmax)*get_alpha_lower(input_spin-1, lmax)) for alms in grad_curl_alms
    ])
    if input_spin - 2 == 0:
        spin_2_lowered_maps = -np.array([_alm2map_ducc0(alms, nside=nside, spin=0, lmax=lmax) for alms in _gclm])
    elif input_spin - 2 < 0:
        spin_2_lowered_maps = _alm2map_ducc0(_gclm, nside=nside, spin=np.abs(input_spin - 2), lmax=lmax)
        
        spin_2_lowered_maps[1] *= -1
    else:
        spin_2_lowered_maps = _alm2map_ducc0(_gclm, nside=nside, spin=input_spin-2, lmax=lmax)

    # Second obtaining the application of two successsive spin-lowering operators on the input alms
    _gclm = [
        hp.almxfl(alms, get_alpha_raise(input_spin, lmax)*get_alpha_raise(input_spin+1, lmax)) for alms in grad_curl_alms
    ]
    spin_2_raised_maps = np.array(hp.alm2map_spin(_gclm, nside, input_spin+2, lmax))

    # Third obtaining the application of the spin-raising then the spin-lowering operators on the input alms
    _gclm = np.array([
        hp.almxfl(alms, get_alpha_raise(input_spin, lmax)*get_alpha_lower(input_spin+1, lmax)) for alms in grad_curl_alms
    ])
    if input_spin == 0:
        spin_raised_lowered_maps = -np.array([_alm2map_ducc0(alms, nside=nside, spin=0, lmax=lmax) for alms in _gclm]).squeeze()
    else:
        spin_raised_lowered_maps = np.array(hp.alm2map_spin(_gclm, nside, input_spin, lmax))

    # Fourth obtaining the application of the spin-lowering then the spin-raising operators on the input alms
    _gclm = np.array([
        hp.almxfl(alms, get_alpha_lower(input_spin, lmax)*get_alpha_raise(input_spin-1, lmax)) for alms in grad_curl_alms
    ])
    if input_spin == 0:
        spin_lowered_raised_maps = -np.array([_alm2map_ducc0(alms, nside=nside, spin=0, lmax=lmax) for alms in _gclm]).squeeze()
    else:
        spin_lowered_raised_maps = np.array(_alm2map_ducc0(_gclm, nside=nside, spin=input_spin, lmax=lmax))

    return {
        input_spin+2: spin_2_raised_maps[0] + 1j * spin_2_raised_maps[1], 
        input_spin-2: spin_2_lowered_maps[0] + 1j * spin_2_lowered_maps[1], 
        '+1-1': spin_lowered_raised_maps[0] + 1j * spin_lowered_raised_maps[1],
        '-1+1': spin_raised_lowered_maps[0] + 1j * spin_raised_lowered_maps[1],
    } 

def get_first_spherical_derivatives_from_spin_derivatives(
        input_map, 
        spin_derivatives_dict,
        nside,
        input_spin,
        zbounds=(-1., 1.),
    ):
    """
    Function to obtain the spherical derivatives from the spin derivatives.

    Parameters
    ----------
    spin_derivatives_dict : dict
        Dictionary containing the spin-s transform of the input spherical harmonic with keys being:
        * '1': map after application of the spin-raising operator on the input alms
        * '-1': map after application of the spin-lowering operator on the input alms
    input_spin : int
        Input spin of the spherical harmonic coefficients grad_curl_alms.
    
    Returns
    -------
    spherical_derivatives_dict : dict
        Dictionary containing the spherical derivatives with keys being:
        * 'theta': first derivative with respect to theta
        * 'phi': first derivative with respect to phi (including factor 1/sin(theta))
    """
    assert input_spin >= 0, input_spin
    assert input_spin+1 in spin_derivatives_dict, f"Spin {input_spin+1} derivative not found in input dictionary, needed for theta and phi derivatives."
    assert input_spin-1 in spin_derivatives_dict, f"Spin {input_spin-1} derivative not found in input dictionary, needed for theta and phi derivatives."

    spherical_derivatives_dict = dict()
    
    # Retrieving the application of the partial derivative over theta from the spin-lowering and spin-raising operators
    spherical_derivatives_dict['theta'] = -0.5 * (spin_derivatives_dict[input_spin+1] + spin_derivatives_dict[input_spin-1])

    # Retrieving the application of the partial derivative over phi (with factor 1/sin(theta)) from the spin-lowering and spin-raising operators
    spherical_derivatives_dict['phi'] = - 0.5 * 1j * (spin_derivatives_dict[input_spin-1] - spin_derivatives_dict[input_spin+1])
    for iring in range(4 * nside - 1):
        startpix, nphi, kphi0, cth, sth = get_healpix_ring_pixel_layout(nside, iring)
        if zbounds[0] <= cth <= zbounds[1]:
            slic = slice(startpix, startpix + nphi)
            spherical_derivatives_dict['phi'][slic] -= 1j * input_spin * (cth / sth) * input_map[slic]

    return spherical_derivatives_dict

def multiply_tan_theta_power(input_map, nside, power=-1, zbounds=(-1., 1.)):
    """
    Function to multiply the input map by (tan(theta))**power.

    Parameters
    ----------
    input_map : np.ndarray
        Input map in healpix format.
    nside : int
        Healpix nside parameter.
    power : int
        Power of the tan(theta) factor to multiply the input map.
    
    Returns
    -------
    output_map : np.ndarray
        Output map in healpix format after multiplication by (tan(theta))**power.
    """
    output_map = np.copy(input_map)
    for iring in range(4 * nside - 1):
        startpix, nphi, kphi0, cth, sth = get_healpix_ring_pixel_layout(nside, iring)
        if zbounds[0] <= cth <= zbounds[1]:
            slic = slice(startpix, startpix + nphi)
            output_map[slic] *= np.where(power>0, sth / cth, cth / sth) ** np.abs(power)
    return output_map

def get_second_spherical_derivatives_from_spin_derivatives(
        input_map,
        spin_derivatives_dict,
        nside,
        input_spin,
        lmax,
        zbounds=(-1., 1.),
        spherical_derivatives_dict: dict = dict()
    ):
    """
    Function to obtain the spherical derivatives from the spin derivatives.

    Parameters
    ----------

    spin_derivatives_dict : dict
        Dictionary containing the spin-s transform of the input spherical harmonic with keys being:
        * '2': map after application of two spin-raising operator on the input alms
        * '-2': map after application of two spin-lowering operator on the input alms
        * '+1-1': map after application of the spin-raising then the spin-lowering operators on the input alms
        * '-1+1': map after application of the spin-lowering then the spin-raising operators on the input alms
    input_spin : int
        Input spin of the spherical harmonic coefficients grad_curl_alms.
    
    Returns
    -------
    spherical_derivatives_dict : dict
        Dictionary containing the spherical derivatives with keys being:
        * 'theta_theta': second derivative with respect to theta 
        * 'phi_phi': second derivative with respect to phi (including factor 1/sin^2(theta))
        * 'theta_phi': mixed second derivative with respect to theta and phi (including factor 1/sin(theta))
        * 'phi': first derivative with respect to phi (including factor 1/sin(theta))
        * 'theta': first derivative with respect to theta
    """

    if 'theta' not in spherical_derivatives_dict and 'phi' not in spherical_derivatives_dict:
        # Computing the first partial derivative with respect to theta
        if input_spin+1 not in spin_derivatives_dict or input_spin-1 not in spin_derivatives_dict:
            input_alms = map2alm_ducc0_iter(input_map, lmax=lmax, spin=0, iter=10)
            spin_derivatives_dict_ = get_first_spin_derivative(
                -np.vstack([input_alms, np.zeros_like(input_alms)]), 
                nside,
                input_spin,
            )
            spin_derivatives_dict[input_spin+1] = spin_derivatives_dict_[input_spin+1]
            spin_derivatives_dict[input_spin-1] = spin_derivatives_dict_[input_spin-1]

        spherical_derivatives_dict_ = get_first_spherical_derivatives_from_spin_derivatives(
            input_map, 
            spin_derivatives_dict,
            nside,
            input_spin,
            zbounds=zbounds,
        )
        spherical_derivatives_dict['theta'] = spherical_derivatives_dict_['theta']
        spherical_derivatives_dict['phi'] = spherical_derivatives_dict_['phi']
    

    assert input_spin+2 in spin_derivatives_dict, f"Spin {input_spin+2} derivative not found in input dictionary, needed for theta_theta, theta_phi and phi_phi second derivatives."
    assert input_spin-2 in spin_derivatives_dict, f"Spin {input_spin-2} derivative not found in input dictionary, needed for theta_theta, theta_phi and phi_phi second derivatives."
    assert '+1-1' in spin_derivatives_dict, f"Spin '+1-1' derivative not found in input dictionary, needed for theta_theta, theta_phi and phi_phi second derivatives."
    assert '-1+1' in spin_derivatives_dict, f"Spin '-1+1' derivative not found in input dictionary, needed for theta_theta, theta_phi and phi_phi second derivatives."


    # Computing the double partial derivative with respect to theta
    spherical_derivatives_dict['theta_theta'] = 0.25 * (
        spin_derivatives_dict[input_spin+2]
        + spin_derivatives_dict[input_spin-2]
        + spin_derivatives_dict['+1-1']
        + spin_derivatives_dict['-1+1']
    )

    # Computing the first term for the partial derivatives with respect to phi and theta with the factor 1/sin(theta) 
    # d_phi_sin0_d_th = 0.25 * (_sp2_d - _sm2_d)
    spherical_derivatives_dict['theta_phi'] = - 0.25 * 1j * (
        spin_derivatives_dict[input_spin+2] - spin_derivatives_dict[input_spin-2]
    )

    # Computing the first term for the double partial derivative with respect to phi with the factor 1/sin(theta)**2
    spherical_derivatives_dict['phi_phi'] = 0.25 * (
        spin_derivatives_dict['+1-1'] 
        + spin_derivatives_dict['-1+1'] 
        - (spin_derivatives_dict[input_spin+2] 
        + spin_derivatives_dict[input_spin-2])
    )

    for iring in range(4 * nside - 1):
        startpix, nphi, kphi0, cth, sth = get_healpix_ring_pixel_layout(nside, iring)
        if zbounds[0] <= cth <= zbounds[1]:
            slic = slice(startpix, startpix + nphi)

            spherical_derivatives_dict['theta_phi'][slic] += 1j * (
                (input_spin * (cth / sth) **2 + input_spin/2.)* input_map[slic]  
                - input_spin * (cth / sth) * spherical_derivatives_dict['theta'][slic]
            ) + ((cth / sth) * spherical_derivatives_dict['phi'][slic])


            spherical_derivatives_dict['phi_phi'][slic] -= (
                - input_spin**2 * (cth / sth) ** 2 * input_map[slic] 
                + (cth / sth) * spherical_derivatives_dict['theta'][slic] 
                + 1j * (2 * input_spin * (cth / sth) * spherical_derivatives_dict['phi'][slic]) 
            )

    return spherical_derivatives_dict
