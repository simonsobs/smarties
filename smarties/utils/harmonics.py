# This file is part of SMARTIES.
# Copyright (C) 2024-2026 Members of the Simons Observatory collaboration.
# Please refer to the LICENSE file in the root of this repository.

from os import process_cpu_count
import numpy as np
import healpy as hp
from numba import njit,prange
from pixell import enmap, curvedsky
import ducc0


__all__ = [
    'map2alm_anypix',
    'alm2map_anypix'
]


def _ducc_dictionary(
        spin: int,
        nside: int,
        lmax: int,
        mmax: int = None) -> dict:
    """
    Create a dictionary of parameters for ducc0 SHT functions.

    Parameters
    ----------
    spin : int
        Spin of the spherical harmonics.
    nside : int
        HEALPix nside parameter.
    lmax : int
        Maximum multipole moment.
    mmax : int, optional
        Maximum order of the spherical harmonics. If None, defaults to lmax.

    Returns
    -------
    dict
        Dictionary containing the parameters for ducc0 healpix SHT functions.
    """

    ducc_healpix_obj = ducc0.healpix.Healpix_Base(nside, 'RING')
    if mmax is None:
        mmax = lmax
    # m_array = np.arange(mmax + 1)
    return {'spin': spin,
              'lmax': lmax,
              'mmax': mmax,
            #   'mstart': (m_array*(2*lmax+1-m_array)//2).astype(np.uint64, copy=False),
              **ducc_healpix_obj.sht_info()
    }

def _alm2map_ducc0(
    alms: np.ndarray, 
    spin: int, 
    nside: int, 
    lmax: int = None, 
    mmax: int = None, 
    nthreads: int = -1
) -> np.ndarray:
    """
    Perform spherical harmonic synthesis using ducc0, to produce healpix maps from spherical harmonic coefficients.
    """

    if nthreads < 0:
        nthreads = process_cpu_count()

    if alms.ndim > 1:
        alm_size = alms.shape[-1]
    else:
        alm_size = alms.size
    if lmax is None:
        lmax = hp.Alm.getlmax(alms.shape[-1])
    else:
        assert lmax <= hp.Alm.getlmax(alms.shape[-1]), (lmax, hp.Alm.getlmax(alms.shape[-1]))
    if mmax is None:
        mmax = lmax

    maps = ducc0.sht.synthesis(
        alm=np.atleast_2d(alms),
        nthreads=nthreads,
        **_ducc_dictionary(
            spin,
            nside,
            lmax,
            mmax,
        )
    )
    return maps


def _map2alm_ducc0(
    maps: np.ndarray, 
    spin: int, 
    lmax: int = None, 
    mmax: int = None, 
    nthreads: int = -1
) -> np.ndarray:
    """
    Perform spherical harmonic analysis using ducc0, from healpix maps. 
    """

    nside = hp.npix2nside(maps.shape[-1])

    if lmax is None:
        lmax = 3 * nside - 1

    if mmax is None:
        mmax = lmax

    if nthreads < 0:
        nthreads = process_cpu_count()

    weight = 4*np.pi/(12 * nside**2)
    alm = ducc0.sht.adjoint_synthesis(
        map=np.atleast_2d(maps) * weight,
        nthreads=nthreads,
        **_ducc_dictionary(
            spin,
            nside,
            lmax,
            mmax,
        )
    )
    return alm

def map2alm_ducc0_iter(
    maps: np.ndarray, 
    spin: int, 
    lmax: int = None, 
    mmax: int = None, 
    niter: int = 3
) -> np.ndarray:
    """
    Perform spherical harmonic analysis using ducc0 with iterative refinement,
    for healpix maps. 

    Parameters
    ----------
    maps : np.ndarray
        Input map(s) in HEALPix format.
    spin : int
        Spin of the spherical harmonics.
    lmax : int, optional
        Maximum multipole moment. If None, defaults to 3*nside-1.
    mmax : int, optional
        Maximum order of the spherical harmonics. If None, defaults to lmax.
    niter : int, optional
        Number of iterations for refinement. Default is 3.

    Returns
    -------
    np.ndarray
        Spherical harmonic coefficients (alm) corresponding to the input map(s).
    """
    nside = hp.npix2nside(maps.shape[-1]) if type(maps) != enmap.ndmap else None

    alms_output = _map2alm_ducc0(maps, spin=spin, lmax=lmax, mmax=mmax)

    for iter_ in range(niter):
        residual_map = _alm2map_ducc0(alms_output, spin=spin, nside=nside, lmax=lmax, mmax=mmax) - maps
        alms_output -= _map2alm_ducc0(residual_map, spin=spin, lmax=lmax, mmax=mmax)
    return alms_output

def map2alm_anypix(
        maps: np.ndarray | enmap.ndmap,
        spin: int,
        niter: int = 0,
        lmax: int = None,
        mmax: int = None,
        shape_car: tuple | None = None
    ) -> np.ndarray:
    """
    Performs map2alm to retrieve SHT coefficients from pixel map, independant
    of the input pixelization (either CAR or HEALPIX).

    Parameters
    __________
    maps: np.ndarray | enmap.ndmap
        Pixel map from which the SHT coefficients will be computed, can be either 
        a CAR map (enmap.ndmap) or a HEALPIX map (np.ndarray).
    spin: int
        Input spin in terms of spin-weighted spherical harmonics
    niter: int
        Number of iterations for refinement (only used for HEALPIX maps)
    lmax: int
        Maximum multipole moment
    mmax: int
        Maximum order of the spherical harmonics
    shape_car: tuple | None
        Shape of the CAR map (only used for CAR maps, if flattened), if None, the shape of the input map will be used.

    Returns
    -------
    np.ndarray
        Spherical harmonic coefficients (alm) corresponding to the input map(s).
    """
    if type(maps) == enmap.ndmap:
        # CAR pixelization expected
        if shape_car is not None:
            assert type(shape_car) == tuple
            maps_to_pass = maps if maps.shape[:-2] == shape_car else maps.reshape(maps.shape[:-1]+shape_car)
        else:
            assert maps.ndim >= 2
            maps_to_pass = maps

        return curvedsky.map2alm(
            maps_to_pass,
            spin=spin,
            lmax=lmax,
            niter=niter
        )
    else:
        # HEALPIX pixelization expected
        return map2alm_ducc0_iter(
            maps=maps,
            spin=spin,
            lmax=lmax,
            mmax=mmax,
            niter=niter
        )

def alm2map_anypix(
        alms: np.ndarray,
        spin: int,
        shape_pixels_output: tuple | int,
        map_output: enmap.ndmap | None = None,
        lmax: int | None = None,
        mmax: int | None = None,
) -> np.ndarray | enmap.ndmap:
    """
    Performs alm2map to retrieve pixel map from SHT coefficients, independant
    of the expected output pixelization (either CAR or HEALPIX).

    Parameters
    __________
    alms: np.ndarray
        Spherical harmonics coefficients from which the pixel map will be
        computed
    spin: int
        Input spin in terms of spin-weighted spherical harmonics
    shape_pixels_output: tuple or int
        Pixel shape of the output pixel map (only the last dimensions of the output),
        the parameter can be either an int or 1d tuple representing an HEALPIX map,
        or a 2d tuple representing a CAR map.
    map_output: enmap.ndmap | None
        Output map to which the pixel map will be written, only used for CAR maps and
        need to be provided if the output pixelization is CAR, otherwise it should be None.
    lmax: int | None
        Maximum multipole moment 
    mmax: int | None
        Maximum order of the spherical harmonics 

    Returns
    -------
    np.ndarray | enmap.ndmap
        Pixel map corresponding to the input SHT coefficients, either in CAR or 
        HEALPIX pixelization depending on the input parameters.
    """

    if map_output is not None:
        # CAR pixelization expected
        assert type(map_output) == enmap.ndmap

        return curvedsky.alm2map(
            alms,
            map=map_output,
            spin=spin,
            copy=True
        ).reshape(map_output.shape[:-2] + (np.prod(shape_pixels_output),))
    else:
        # HEALPIX pixelization expected
        if spin == 0:
            return np.array([
                _alm2map_ducc0(
                alms=_alm,
                spin=spin,
                nside=hp.npix2nside(np.prod(shape_pixels_output)),
                lmax=lmax,
                mmax=mmax,
                )
            for _alm in (alms if alms.ndim > 1 else [alms])
        ])

        return _alm2map_ducc0(
            alms=alms,
            spin=spin,
            nside=hp.npix2nside(np.prod(shape_pixels_output)),
            lmax=lmax,
            mmax=mmax,
        )

def convert_alm_plusminus_to_spin(
    alm_plus: np.ndarray, alm_minus: np.ndarray, spin: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert +/- basis alms coefficients to spin-weighted alms.

    Parameters
    ----------
    alm_plus: np.ndarray
        + basis coefficients.
    alm_minus: np.ndarray
        - basis coefficients (same shape as ``alm_plus``).
    spin: int (optional)
        Target spin (default 2).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(alm_pos_spin, alm_neg_spin)`` following Healpix convention
    """
    alms_pos_spin = -1 * (alm_plus + 1j * alm_minus)  # |spin| component
    alms_neg_spin = (alm_plus - 1j * alm_minus) * (-1.0) ** (
        -1 - spin
    )  # -|spin| component

    return alms_pos_spin, alms_neg_spin


def convert_alm_spin_to_plusminus(
    alm_pos_spin: np.ndarray, alm_neg_spin: np.ndarray, spin: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert spin-weighted alms coefficients to the +/- basis.

    Parameters
    ----------
    alm_pos_spin: np.ndarray
        +spin coefficients.
    alm_neg_spin: np.ndarray
        -spin coefficients (same shape as ``alm_pos_spin``).
    spin: int (optional)
        Spin of the input coefficients (default 2).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(alm_plus, alm_minus)`` following Healpix convention
    """

    alm_plus = -1 * (alm_pos_spin + (-1) ** (spin) * alm_neg_spin) / (2)
    alm_minus = -1 * (alm_pos_spin - (-1) ** (spin) * alm_neg_spin) / (2j)
    return alm_plus, alm_minus

def almxfl(alm:np.ndarray, fl:np.ndarray, mmax:int|None=None) -> np.ndarray:
    """
    Multiply alm by a function of ell, f(ell)

    Parameters
    ----------
    alm : array_like
        Input alm array.
    fl : array_like
        Function of ell to multiply the alm by.
    mmax : int
        Maximum m of the alm array, if None assumes mmax=lmax.

    Returns
    -------
    array_like
        Output alm array after multiplication.
    
    Notes
    -----
    This function mimics the behavior of healpy's almxfl, but uses numba for parallel processing,
    and allowing the alms to be kept in single-precision or double-precision, depending on the input alm type.
    """
    lmax= hp.Alm.getlmax(alm.size,mmax)
    assert lmax != -1, "Invalid alm size for given mmax"
    if mmax is None:
        mmax= lmax
    flsize= fl.size
    alm_out=np.empty_like(alm)
    _compute_almxfl_prange(alm=alm,alm_out=alm_out, fl=fl,flsize=flsize,lmax=lmax, mmax=mmax)
    # _compute_almxfl(alm=alm,alm_out=alm_out, fl=fl,flsize=flsize,lmax=lmax, mmax=mmax)
    # Multiply by fl using healpy's almxfl
    return alm_out



@njit(parallel=True)
def _compute_almxfl_prange(alm: np.ndarray, alm_out: np.ndarray, fl: np.ndarray, flsize: int, lmax: int, mmax: int) -> np.ndarray:
    """
    Multiply alm by a function of ell, f(ell) using parallel processing.
    """
    for m in prange(mmax+ 1): # type: ignore[not-iterable]

        idx_m=m * (2 * lmax + 1 - m) // 2
        for l in range(m, lmax + 1):
            idx_lm= idx_m + l
            if l < flsize:
                alm_out[idx_lm] =alm[idx_lm]* fl[l]
            else:
                alm_out[idx_lm] = 0.
    return alm_out
