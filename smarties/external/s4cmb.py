# S4CMB
# Copyright (c) 2016-2021, 2025-2026 Julien Peloton, Giulio Fabbian, Magdy Morshed
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
Routines taken from s4cmb and readapted including diverse tools to manipulate alms and maps from the s4cmb package.
"""

import healpy as hp
import numpy as np
from pixell import enmap, wcsutils

from smarties.utils.harmonics import alm2map_anypix, map2alm_anypix

__all__ = [
    "get_healpix_ring_pixel_layout",
    "get_car_ring_layout",
    "compute_phi_1st_derivative",
    "compute_phi_2nd_derivatives",
    "multiply_tan_theta_power",
    "get_alpha_raise",
    "get_alpha_lower",
    "get_first_spin_derivative",
    "get_second_spin_derivative",
    "get_first_spherical_derivatives_from_spin_derivatives",
    "get_second_spherical_derivatives_from_spin_derivatives",
]


def get_healpix_ring_pixel_layout(nside: int, th_idx: int) -> tuple[int, int, int, float, float]:
    """Healpix ring layout.

    From 'get_pixel_layout' subroutine in healpix f90 package.
    Given an nside and the ring index, returns the starting pixel number, 
    number of pixels in the ring, starting pixel number in the ring, cosine and sine of the theta polar angle.

    Original author: Julien Carron (j.carron@sussex.ac.uk)

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
        return 12 * nside**2 - startpix - nphi, nphi, kphi0, -cth, sth
    dth1 = 1.0 / 3.0 / nside**2
    dth2 = 2.0 / 3.0 / nside
    dst1 = 1.0 / (np.sqrt(6.0) * nside)
    if ith < nside:  # polar cap (north)
        cth = 1.0 - ith**2 * dth1
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


def get_car_ring_layout(shape: tuple[int], wcs: wcsutils.WCS) -> tuple[np.ndarray, np.ndarray]:
    """Get the layout of the CAR pixelization, returning the cosine and sine of the theta polar angle for each pixel.

    Parameters
    ----------
    shape : tuple[int]
        Shape of the CAR map (ny, nx).
    wcs : wcsutils.WCS
        WCS object for the CAR map.

    Returns
    -------
    cth : np.ndarray
        Cosine of the polar angle for each pixel.
    sth : np.ndarray
        Sine of the polar angle for each pixel.
    """
    pixel_coordinates_th = (
        np.pi / 2
        - enmap.pix2sky(
            shape,
            wcs,
            np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij"),
        )[0]
    ) % (2 * np.pi)

    sth = np.sin(pixel_coordinates_th)
    cth = np.cos(pixel_coordinates_th)
    return cth, sth


def compute_phi_1st_derivative(
    input_map: np.ndarray | enmap.ndmap,
    spin_derivatives_dict: dict[int, np.ndarray],
    input_spin: int,
    shape_pixels_output: tuple[int, int] | None = None,
    zbounds: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray | enmap.ndmap:
    """
    Compute the first derivative of the input map with respect to phi, using the spin-raising and spin-lowering operators.

    Parameters
    ----------
    input_map : np.ndarray | enmap.ndmap
        Input map in either CAR or HEALPIX format.
    spin_derivatives_dict : dict
        Dictionary containing the spin derivatives of the input map:
        * 'input_spin + 1': map after application of the spin-raising operator on the input map
        * 'input_spin - 1': map after application of the spin-lowering operator on the input map
    input_spin : int
        The spin of the input map.
    shape_pixels_output : tuple[int], optional
        Shape of the output map in pixels (ny, nx). If None, it will be inferred from the input_map.
    zbounds : tuple[float], optional
        The bounds for the z-coordinate (cosine of the polar angle) to consider (default to full sky).

    Returns
    -------
    spherical_derivatives_phi : np.ndarray | enmap.ndmap
        The first derivative of the input map with respect to phi.
    """
    if type(input_map) == enmap.ndmap:
        if input_map.ndim == 1:
            assert len(shape_pixels_output) == 2, (
                "The shape_car parameter must be provided and have length 2 for flattened input_map."
            )
        elif input_map.ndim >= 2:
            assert (
                shape_pixels_output is None
                or shape_pixels_output == input_map.shape[-2:]
            ), "The shape_car parameter must be None or match the shape of input_map."
            shape_pixels_output = input_map.shape[-2:]

        cth, sth = get_car_ring_layout(shape_pixels_output, input_map.wcs)
        cth = cth.ravel()
        sth = sth.ravel()

        spherical_derivatives_phi = -1j * (
            0.5
            * (
                spin_derivatives_dict[input_spin - 1]
                - spin_derivatives_dict[input_spin + 1]
            )
            + input_spin * (cth / sth) * input_map
        )

    else:
        nside = hp.npix2nside(input_map.shape[-1])

        spherical_derivatives_phi = (
            -0.5
            * 1j
            * (
                spin_derivatives_dict[input_spin - 1]
                - spin_derivatives_dict[input_spin + 1]
            )
        )

        for iring in range(4 * nside - 1):
            startpix, nphi, kphi0, cth, sth = get_healpix_ring_pixel_layout(
                nside, iring
            )
            if zbounds[0] <= cth <= zbounds[1]:
                slic = slice(startpix, startpix + nphi)
                spherical_derivatives_phi[slic] -= (
                    1j * input_spin * (cth / sth) * input_map[slic]
                )
    return spherical_derivatives_phi


def compute_phi_2nd_derivatives(
    input_map: np.ndarray | enmap.ndmap,
    spin_derivatives_dict: dict[int, np.ndarray],
    spherical_derivatives_dict: dict[str, np.ndarray],
    input_spin: int,
    shape_car: tuple[int, int] | None = None,
    zbounds: tuple[float, float] = (-1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray] | tuple[enmap.ndmap, enmap.ndmap]:
    """
    Compute the second derivatives of the input map with respect to phi, using the spin-raising and spin-lowering operators.

    Parameters
    ----------
    input_map : np.ndarray | enmap.ndmap
        Input map in either CAR or HEALPIX format.
    spin_derivatives_dict : dict
        Dictionary containing the spin derivatives of the input map:
        * 'input_spin + 2': map after double application of the spin-raising operator on the input map
        * 'input_spin - 2': map after double application of the spin-lowering operator on the input map
    spherical_derivatives_dict : dict
        Dictionary containing the spherical derivatives of the input map:
        * 'theta': first derivative of the input map with respect to theta
        * 'phi': first derivative of the input map with respect to phi
    input_spin : int
        The spin of the input map.
    shape_car : tuple[int], optional
        Shape of the output map in pixels (ny, nx). If None, it will be inferred from the input_map.
    zbounds : tuple[float], optional
        The bounds for the z-coordinate (cosine of the polar angle) to consider (default to full sky).

    Returns
    -------
    spherical_derivatives_theta_phi : np.ndarray | enmap.ndmap
        The second derivative of the input map with respect to theta and phi.
    spherical_derivatives_phi_phi : np.ndarray | enmap.ndmap
        The second derivative of the input map with respect to phi.
    """
    spherical_derivatives_theta_phi = (
        -0.25
        * 1j
        * (
            spin_derivatives_dict[input_spin + 2]
            - spin_derivatives_dict[input_spin - 2]
        )
    )

    # Computing the first term for the double partial derivative with respect to phi with the factor 1/sin(theta)**2
    spherical_derivatives_phi_phi = 0.25 * (
        spin_derivatives_dict["+1-1"]
        + spin_derivatives_dict["-1+1"]
        - (
            spin_derivatives_dict[input_spin + 2]
            + spin_derivatives_dict[input_spin - 2]
        )
    )

    if type(input_map) == enmap.ndmap:
        if input_map.ndim == 1:
            assert len(shape_car) == 2, (
                "The shape_car parameter must be provided and have length 2 for flattened input_map."
            )
        elif input_map.ndim >= 2:
            assert shape_car is None or shape_car == input_map.shape[-2:], (
                "The shape_car parameter must be None or match the shape of input_map."
            )
            shape_car = input_map.shape[-2:]
        cth, sth = get_car_ring_layout(shape_car, input_map.wcs)
        cth = cth.ravel()
        sth = sth.ravel()

        spherical_derivatives_theta_phi += 1j * (
            (input_spin * (cth / sth) ** 2 + input_spin / 2.0) * input_map
            - input_spin * (cth / sth) * spherical_derivatives_dict["theta"]
        ) + ((cth / sth) * spherical_derivatives_dict["phi"])

        spherical_derivatives_phi_phi -= (
            -(input_spin**2) * (cth / sth) ** 2 * input_map
            + (cth / sth) * spherical_derivatives_dict["theta"]
            + 1j * (2 * input_spin * (cth / sth) * spherical_derivatives_dict["phi"])
        )

    else:
        nside = hp.npix2nside(input_map.shape[-1])
        for iring in range(4 * nside - 1):
            startpix, nphi, kphi0, cth, sth = get_healpix_ring_pixel_layout(
                nside, iring
            )
            if zbounds[0] <= cth <= zbounds[1]:
                slic = (..., slice(startpix, startpix + nphi))

                spherical_derivatives_theta_phi[slic] += 1j * (
                    (input_spin * (cth / sth) ** 2 + input_spin / 2.0) * input_map[slic]
                    - input_spin
                    * (cth / sth)
                    * spherical_derivatives_dict["theta"][slic]
                ) + ((cth / sth) * spherical_derivatives_dict["phi"][slic])

                spherical_derivatives_phi_phi[slic] -= (
                    -(input_spin**2) * (cth / sth) ** 2 * input_map[slic]
                    + (cth / sth) * spherical_derivatives_dict["theta"][slic]
                    + 1j
                    * (
                        2
                        * input_spin
                        * (cth / sth)
                        * spherical_derivatives_dict["phi"][slic]
                    )
                )
    return spherical_derivatives_theta_phi, spherical_derivatives_phi_phi


def multiply_tan_theta_power(
    input_map: np.ndarray | enmap.ndmap,
    power: int = -1,
    zbounds: tuple[float, float] = (-1.0, 1.0),
    shape_car: tuple[int, int] | None = None,
) -> np.ndarray | enmap.ndmap:
    """
    Function to multiply the input map by (tan(theta))**power.

    Parameters
    ----------
    input_map : np.ndarray | enmap.ndmap
        Input map in either CAR or HEALPIX format.
    power : int
        Power of the tan(theta) factor to multiply the input map.
    zbounds : tuple[float], optional
        The bounds for the z-coordinate (cosine of the polar angle) to consider (default to full sky).
    shape_car : tuple[int], optional
        Shape of the output map in pixels (ny, nx), if the input_map is in CAR format. 
        If None, it will be inferred from the input_map.

    Returns
    -------
    output_map : np.ndarray | enmap.ndmap
        Output map after multiplication by (tan(theta))**power.
    """

    output_map = input_map.copy()
    if type(input_map) == enmap.ndmap:
        if shape_car is not None:
            assert len(shape_car) == 2, (
                "The shape_car parameter must be provided and have length 2 for flattened input_map."
            )
        elif input_map.ndim >= 2:
            assert shape_car is None or shape_car == input_map.shape[-2:], (
                "The shape_car parameter must be None or match the shape of input_map."
            )
            shape_car = input_map.shape[-2:]
        else:
            raise ValueError(
                "The shape_car parameter must be provided if the input_map is 1D."
            )
        cth, sth = get_car_ring_layout(shape_car, input_map.wcs)
        cth = cth.ravel()
        sth = sth.ravel()

        return enmap.ndmap(
            output_map * (np.where(power > 0, sth / cth, cth / sth) ** np.abs(power)),
            wcs=input_map.wcs,
        )

    else:
        nside = hp.npix2nside(input_map.shape[-1])
        for iring in range(4 * nside - 1):
            startpix, nphi, kphi0, cth, sth = get_healpix_ring_pixel_layout(
                nside, iring
            )
            if zbounds[0] <= cth <= zbounds[1]:
                slic = slice(startpix, startpix + nphi)
                output_map[slic] *= np.where(power > 0, sth / cth, cth / sth) ** np.abs(
                    power
                )
    return output_map



def get_alpha_raise(spin: int, lmax: int) -> np.ndarray:
    """Response coefficient of spin-s spherical harmonic to spin raising operator.

    Author: Julien Carron (j.carron@sussex.ac.uk)

    Parameters
    ----------
    spin : int
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
        alpha(spin, ell) = sqrt((ell - spin) * (ell + spin + 1))
    where ell is the multipole moment.
    The response coefficient is zero for ell < |spin|.
    """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(spin) :] = np.sqrt(
        np.arange(abs(spin) - spin, lmax - spin + 1) * np.arange(abs(spin) + spin + 1, lmax + spin + 2)
    )
    return ret



def get_alpha_lower(spin: int, lmax: int) -> np.ndarray:
    """Response coefficient of spin-s spherical harmonic to spin lowering operator.

    Author: Julien Carron (j.carron@sussex.ac.uk)

    Parameters
    ----------
    spin : int
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
        alpha(spin, ell) = sqrt((ell + spin) * (ell - spin + 1))
    where ell is the multipole moment.
    The response coefficient is zero for ell < |spin|.
    """
    ret = np.zeros(lmax + 1, dtype=float)
    ret[abs(spin) :] = -np.sqrt(
        np.arange(spin + abs(spin), lmax + spin + 1) * np.arange(abs(spin) - spin + 1, lmax - spin + 2)
    )
    return ret


def get_first_spin_derivative(
    grad_curl_alms: list[np.ndarray], 
    shape_pixels_output: tuple, 
    input_spin: int, 
    wcs=None
) -> dict[int, np.ndarray | enmap.ndmap]:
    """
    Function to obtain the maps after applying the spin-raising and spin-lowering operators on the input alms of arbitrary spin.

    Parameters
    ----------
    grad_curl_alms : list of np.ndarray
        List of two arrays containing the grad and curl parts of the spherical harmonic coefficients.
        grad_curl_alms[0] is the grad part and grad_curl_alms[1] is the curl part, and the last dimension is the healpix ordering
        of the alms.
    shape_pixels_output : tuple or int
        Pixel shape of the output pixel map.
    input_spin : int
        Input spin of the spherical harmonic coefficients grad_curl_alms.

    Returns
    -------
    dictionary_spin_derivative : dict
        Dictionary containing the spin-s transform of the input spherical harmonic with keys being:
        * input_spin + 1: map after application of the spin-raising operator on the input alms
        * input_spin + -1: map after application of the spin-lowering operator on the input alms

    """

    assert input_spin >= 0, input_spin
    assert hp.Alm.getlmax(grad_curl_alms[0].size) == hp.Alm.getlmax(
        grad_curl_alms[1].size
    )
    lmax = hp.Alm.getlmax(grad_curl_alms[0].size)
    # shape (2, 12 * nside ** 2),

    map_output = (
        enmap.empty(tuple(grad_curl_alms.shape[:-1]) + shape_pixels_output, wcs=wcs)
        if wcs is not None
        else None
    )

    # First obtaining the application of the spin-lowering operator on the input alms
    _gclm = np.array(
        [
            hp.almxfl(grad_curl_alms[0], get_alpha_raise(input_spin, lmax)),
            hp.almxfl(grad_curl_alms[1], get_alpha_raise(input_spin, lmax)),
        ]
    )
    spin_raised_maps = alm2map_anypix(
        _gclm,
        map_output=map_output,
        spin=input_spin + 1,
        lmax=lmax,
        shape_pixels_output=shape_pixels_output,
    )

    # Second obtaining the application of the spin-raising operator on the input alms
    if input_spin == 0:
        spin_lowered_maps = np.copy(spin_raised_maps)
        spin_lowered_maps[1] *= -1
    else:
        _gclm = np.array(
            [
                hp.almxfl(grad_curl_alms[0], get_alpha_lower(input_spin, lmax)),
                hp.almxfl(grad_curl_alms[1], get_alpha_lower(input_spin, lmax)),
            ]
        )
        spin_lowered_maps = alm2map_anypix(
            _gclm,
            map_output=map_output,
            spin=input_spin - 1,
            lmax=lmax,
            shape_pixels_output=shape_pixels_output,
        )

    return {
        input_spin + 1: spin_raised_maps[0] + 1j * spin_raised_maps[1],
        input_spin - 1: spin_lowered_maps[0] + 1j * spin_lowered_maps[1],
    }


def get_second_spin_derivative(
    grad_curl_alms: list[np.ndarray], 
    shape_pixels_output: tuple, 
    input_spin: int, 
    wcs: wcsutils.WCS | None = None
) -> dict[int, np.ndarray | enmap.ndmap]:
    """
    Function to obtain the maps after applying any combination of two applications of the spin-raising and spin-lowering operators on the input alms of arbitrary spin.

    Parameters
    ----------
    grad_curl_alms : list of np.ndarray
        List of two arrays containing the grad and curl parts of the spherical harmonic coefficients.
        grad_curl_alms[0] is the grad part and grad_curl_alms[1] is the curl part
    shape_pixels_output : tuple
        Pixel shape of the output pixel map.
    input_spin : int
        Input spin of the spherical harmonic coefficients grad_curl_alms.
    wcs : wcsutils.WCS, optional
        WCS object for the output map. If None, the output will be in HEALPIX format.

    Returns
    -------
    dictionary_spin_derivative : dict
        Dictionary containing the spin-s transform of the input spherical harmonic with keys being:
        * input_spin + 2: map after application of two spin-raising operator on the input alms
        * input_spin + -2: map after application of two spin-lowering operator on the input alms
        * '+1-1': map after application of the spin-raising then the spin-lowering operators on the input alms
        * '-1+1': map after application of the spin-lowering then the spin-raising operators on the input alms

    """

    assert input_spin >= 0, input_spin
    assert hp.Alm.getlmax(grad_curl_alms[0].size) == hp.Alm.getlmax(
        grad_curl_alms[1].size
    )
    lmax = hp.Alm.getlmax(grad_curl_alms[0].size)
    # shape (2, 12 * nside ** 2),

    map_output = None
    if len(shape_pixels_output) == 2 and wcs is not None:
        # CAR pixelization expected
        map_output = enmap.empty(
            tuple(grad_curl_alms.shape[:-1]) + shape_pixels_output, wcs=wcs
        )

    # First obtaining the application of two successsive spin-raising operators on the input alms
    _gclm = np.array(
        [
            hp.almxfl(
                alms,
                get_alpha_lower(input_spin, lmax)
                * get_alpha_lower(input_spin - 1, lmax),
            )
            for alms in grad_curl_alms
        ]
    )
    if input_spin - 2 == 0:
        spin_2_lowered_maps = alm2map_anypix(
            _gclm,
            map_output=map_output,
            shape_pixels_output=shape_pixels_output,
            spin=0,
            lmax=lmax,
        )
    elif input_spin - 2 < 0:
        spin_2_lowered_maps = alm2map_anypix(
            _gclm,
            map_output=map_output,
            shape_pixels_output=shape_pixels_output,
            spin=np.abs(input_spin - 2),
            lmax=lmax,
        )

        spin_2_lowered_maps[1] *= -1
    else:
        spin_2_lowered_maps = alm2map_anypix(
            _gclm,
            map_output=map_output,
            shape_pixels_output=shape_pixels_output,
            spin=input_spin - 2,
            lmax=lmax,
        )

    # Second obtaining the application of two successsive spin-lowering operators on the input alms
    _gclm = np.array(
        [
            hp.almxfl(
                alms,
                get_alpha_raise(input_spin, lmax)
                * get_alpha_raise(input_spin + 1, lmax),
            )
            for alms in grad_curl_alms
        ]
    )
    spin_2_raised_maps = alm2map_anypix(
        _gclm,
        map_output=map_output,
        shape_pixels_output=shape_pixels_output,
        spin=input_spin + 2,
        lmax=lmax,
    )

    # Third obtaining the application of the spin-raising then the spin-lowering operators on the input alms
    _gclm = np.array(
        [
            hp.almxfl(
                alms,
                get_alpha_raise(input_spin, lmax)
                * get_alpha_lower(input_spin + 1, lmax),
            )
            for alms in grad_curl_alms
        ]
    )
    if input_spin == 0:
        spin_raised_lowered_maps = alm2map_anypix(
            _gclm,
            map_output=map_output,
            shape_pixels_output=shape_pixels_output,
            spin=0,
            lmax=lmax,
        )
    else:
        spin_raised_lowered_maps = alm2map_anypix(
            _gclm,
            map_output=map_output,
            shape_pixels_output=shape_pixels_output,
            spin=input_spin,
            lmax=lmax,
        )

    # Fourth obtaining the application of the spin-lowering then the spin-raising operators on the input alms
    _gclm = np.array(
        [
            hp.almxfl(
                alms,
                get_alpha_lower(input_spin, lmax)
                * get_alpha_raise(input_spin - 1, lmax),
            )
            for alms in grad_curl_alms
        ]
    )
    if input_spin == 0:
        spin_lowered_raised_maps = alm2map_anypix(
            _gclm,
            map_output=map_output,
            shape_pixels_output=shape_pixels_output,
            spin=0,
            lmax=lmax,
        )
    else:
        spin_lowered_raised_maps = alm2map_anypix(
            _gclm,
            map_output=map_output,
            shape_pixels_output=shape_pixels_output,
            spin=input_spin,
            lmax=lmax,
        )

    return {
        input_spin + 2: spin_2_raised_maps[0] + 1j * spin_2_raised_maps[1],
        input_spin - 2: spin_2_lowered_maps[0] + 1j * spin_2_lowered_maps[1],
        "+1-1": spin_lowered_raised_maps[0] + 1j * spin_lowered_raised_maps[1],
        "-1+1": spin_raised_lowered_maps[0] + 1j * spin_raised_lowered_maps[1],
    }


def get_first_spherical_derivatives_from_spin_derivatives(
    input_map: np.ndarray | enmap.ndmap,
    spin_derivatives_dict: dict[int, np.ndarray | enmap.ndmap],
    shape_pixels_output: tuple,
    input_spin: int,
    zbounds: tuple[float, float] = (-1.0, 1.0),
) -> dict[str, np.ndarray | enmap.ndmap]:
    """
    Function to obtain the spherical derivatives from the spin derivatives.

    Parameters
    ----------
    input_map : np.ndarray | enmap.ndmap
        Input map in either CAR or HEALPIX format.
    spin_derivatives_dict : dict
        Dictionary containing the spin-s transform of the input spherical harmonic with keys being:
        * input_spin + 1: map after application of the spin-raising operator on the input alms
        * input_spin - 1: map after application of the spin-lowering operator on the input alms
    shape_pixels_output : tuple
        Pixel shape of the output pixel map.
    input_spin : int
        Input spin of the spherical harmonic coefficients grad_curl_alms.
    zbounds : tuple[float, float], optional
        The bounds for the z-coordinate (cosine of the polar angle) to consider (default to full sky).

    Returns
    -------
    spherical_derivatives_dict : dict
        Dictionary containing the spherical derivatives with keys being:
        * 'theta': first derivative with respect to theta
        * 'phi': first derivative with respect to phi (including factor 1/sin(theta))
    """
    assert input_spin >= 0, input_spin
    assert input_spin + 1 in spin_derivatives_dict, (
        f"Spin {input_spin + 1} derivative not found in input dictionary, needed for theta and phi derivatives."
    )
    assert input_spin - 1 in spin_derivatives_dict, (
        f"Spin {input_spin - 1} derivative not found in input dictionary, needed for theta and phi derivatives."
    )

    spherical_derivatives_dict = dict()

    # Retrieving the application of the partial derivative over theta from the spin-lowering and spin-raising operators
    spherical_derivatives_dict["theta"] = -0.5 * (
        spin_derivatives_dict[input_spin + 1] + spin_derivatives_dict[input_spin - 1]
    )

    # Retrieving the application of the partial derivative over phi (with factor 1/sin(theta)) from the spin-lowering and spin-raising operators

    spherical_derivatives_dict["phi"] = compute_phi_1st_derivative(
        input_map=input_map,
        spin_derivatives_dict=spin_derivatives_dict,
        input_spin=input_spin,
        shape_pixels_output=shape_pixels_output,
        zbounds=zbounds,
    )

    return spherical_derivatives_dict


def get_second_spherical_derivatives_from_spin_derivatives(
    input_map: np.ndarray | enmap.ndmap,
    spin_derivatives_dict: dict,
    shape_pixels_output: tuple,
    input_spin: int,
    lmax: int,
    zbounds: tuple[float, float] = (-1.0, 1.0),
    spherical_derivatives_dict: dict = dict(),
    niter: int =10,
) -> dict[str, np.ndarray | enmap.ndmap]:
    """
    Function to obtain the spherical derivatives from the spin derivatives.

    Parameters
    ----------
    input_map : np.ndarray | enmap.ndmap
        Input map in either CAR or HEALPIX format.
    spin_derivatives_dict : dict
        Dictionary containing the spin-s transform of the input spherical harmonic with keys being:
        * input_spin+2: map after application of two spin-raising operator on the input alms
        * input_spin-2: map after application of two spin-lowering operator on the input alms
        * '+1-1': map after application of the spin-raising then the spin-lowering operators on the input alms
        * '-1+1': map after application of the spin-lowering then the spin-raising operators on the input alms
        If spherical_derivatives_dict is not provided, then spin_derivatives_dict will recompute the keys
        input_spin+2 and input_spin-2, which are needed for the second derivatives.
    shape_pixels_output : tuple
        Pixel shape of the output pixel map.
    input_spin : int
        Input spin of the spherical harmonic coefficients grad_curl_alms.
    lmax : int
        Maximum multipole moment.
    zbounds : tuple[float, float], optional
        The bounds for the z-coordinate (cosine of the polar angle) to consider (default to full sky).
    spherical_derivatives_dict : dict, optional
        Dictionary containing the spherical derivatives with keys being:
        * 'theta': first derivative with respect to theta
        * 'phi': first derivative with respect to phi (including factor 1/sin(theta))
        If not provided, it will be computed from the input map. 
    niter : int, optional
        Number of iterations for the map2alm function (default is 10, only relevant for healpix).

    Returns
    -------
    spherical_derivatives_dict : dict
        The same dictionary spherical_derivatives_dict as the input,
        but with additional keys for the second derivatives, containing the spherical derivatives with keys being:
        * 'theta_theta': second derivative with respect to theta
        * 'phi_phi': second derivative with respect to phi (including factor 1/sin^2(theta))
        * 'theta_phi': mixed second derivative with respect to theta and phi (including factor 1/sin(theta))
        * 'phi': first derivative with respect to phi (including factor 1/sin(theta)) 
        * 'theta': first derivative with respect to theta
    """

    if (
        "theta" not in spherical_derivatives_dict
        and "phi" not in spherical_derivatives_dict
    ):
        # Computing the first partial derivative with respect to theta
        if (
            input_spin + 1 not in spin_derivatives_dict
            or input_spin - 1 not in spin_derivatives_dict
        ):
            input_alms = map2alm_anypix(
                input_map, lmax=lmax, spin=0, niter=niter, shape_car=shape_pixels_output
            )

            wcs = input_map.wcs if type(input_map) == enmap.ndmap else None
            spin_derivatives_dict_ = get_first_spin_derivative(
                grad_curl_alms=-np.vstack([input_alms, np.zeros_like(input_alms)]),
                shape_pixels_output=shape_pixels_output,
                input_spin=input_spin,
                wcs=wcs,
            )
            spin_derivatives_dict[input_spin + 1] = spin_derivatives_dict_[
                input_spin + 1
            ]
            spin_derivatives_dict[input_spin - 1] = spin_derivatives_dict_[
                input_spin - 1
            ]

        spherical_derivatives_dict_ = get_first_spherical_derivatives_from_spin_derivatives(
                input_map=input_map,
                spin_derivatives_dict=spin_derivatives_dict,
                shape_pixels_output=shape_pixels_output,
                input_spin=input_spin,
                zbounds=zbounds,
            )
        spherical_derivatives_dict["theta"] = spherical_derivatives_dict_["theta"]
        spherical_derivatives_dict["phi"] = spherical_derivatives_dict_["phi"]

    assert input_spin + 2 in spin_derivatives_dict, (
        f"Spin {input_spin + 2} derivative not found in input dictionary, needed for theta_theta, theta_phi and phi_phi second derivatives."
    )
    assert input_spin - 2 in spin_derivatives_dict, (
        f"Spin {input_spin - 2} derivative not found in input dictionary, needed for theta_theta, theta_phi and phi_phi second derivatives."
    )
    assert "+1-1" in spin_derivatives_dict, (
        f"Spin '+1-1' derivative not found in input dictionary, needed for theta_theta, theta_phi and phi_phi second derivatives."
    )
    assert "-1+1" in spin_derivatives_dict, (
        f"Spin '-1+1' derivative not found in input dictionary, needed for theta_theta, theta_phi and phi_phi second derivatives."
    )

    # Computing the double partial derivative with respect to theta
    spherical_derivatives_dict["theta_theta"] = 0.25 * (
        spin_derivatives_dict[input_spin + 2]
        + spin_derivatives_dict[input_spin - 2]
        + spin_derivatives_dict["+1-1"]
        + spin_derivatives_dict["-1+1"]
    )

    # Computing the first term for the partial derivatives with respect to phi and theta with the factor 1/sin(theta)
    # d_phi_sin0_d_th = 0.25 * (_sp2_d - _sm2_d)
    spherical_derivatives_dict["theta_phi"] = (
        -0.25
        * 1j
        * (
            spin_derivatives_dict[input_spin + 2]
            - spin_derivatives_dict[input_spin - 2]
        )
    )

    # Computing the first term for the double partial derivative with respect to phi with the factor 1/sin(theta)**2
    spherical_derivatives_dict["phi_phi"] = 0.25 * (
        spin_derivatives_dict["+1-1"]
        + spin_derivatives_dict["-1+1"]
        - (
            spin_derivatives_dict[input_spin + 2]
            + spin_derivatives_dict[input_spin - 2]
        )
    )
    spherical_derivatives_dict["theta_phi"], spherical_derivatives_dict["phi_phi"] = compute_phi_2nd_derivatives(
            input_map=input_map,
            spin_derivatives_dict=spin_derivatives_dict,
            spherical_derivatives_dict=spherical_derivatives_dict,
            input_spin=input_spin,
            shape_car=shape_pixels_output,
            zbounds=zbounds,
        )

    return spherical_derivatives_dict
