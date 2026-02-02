from os import cpu_count
import numpy as np
import healpy as hp
from pixell import enmap, curvedsky
import ducc0



def _ducc_dictionary(
        spin, 
        nside, 
        lmax, 
        mmax=None):
    
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
        **_ducc_dictionary(
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
        **_ducc_dictionary(
            spin, 
            nside, 
            lmax, 
            mmax, 
        )
    )
    return alm

def map2alm_ducc0_iter(maps, spin, lmax=None, mmax=None, niter=3):
    nside = hp.npix2nside(maps.shape[-1]) if type(maps) == enmap.ndmap else None

    alms_output = _map2alm_ducc0(maps, spin=spin, lmax=lmax, mmax=mmax)

    for iter_ in range(niter):
        residual_map = _alm2map_ducc0(alms_output, spin=spin, nside=nside, lmax=lmax, mmax=mmax) - maps
        alms_output -= _map2alm_ducc0(residual_map, spin=spin, lmax=lmax, mmax=mmax)
    return alms_output

def map2alm_anypix(
        maps, 
        spin, 
        niter=0,
        lmax=None, 
        mmax=None,
        shape_car=None
    ):
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
        alms,
        spin, 
        map_output=None,
        shape_pixels_output=None,
        lmax=None, 
        mmax=None,
):
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
        return _alm2map_ducc0(
            alms=alms, 
            spin=spin,
            nside=hp.npix2nside(np.prod(shape_pixels_output)),
            lmax=lmax, 
            mmax=mmax, 
        )
