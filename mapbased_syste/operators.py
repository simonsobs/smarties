import numpy as np
import healpy as hp
from mapbased_syste.hn import Spin_maps

def get_naive_spin_derivative(
        initial_map, 
        input_spin, 
        lmax=None,
        precomputed_sin_theta=None):
    """
    """

    
    if initial_map.ndim > 1:
        raise NotImplementedError("Only 1D maps are supported for now")

    nside = hp.npix2nside(initial_map.size)

    if input_spin == 0:
        sin_theta = 1
        alm_spin_raising = hp.map2alm(
            initial_map, 
            lmax=lmax
        )
        
        
        _, map_dtheta_spin_raising, map_dphi_spin_raising = hp.alm2map_der1(alm_spin_raising, nside, lmax=lmax) 
        # map_dphi already contains the sin(theta) factor

        map_dtheta_spin_lowering, map_dphi_spin_lowering = map_dtheta_spin_raising, map_dphi_spin_raising
        
    else:
        if precomputed_sin_theta is None:
            sin_theta = np.sin(hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))[0])
        else:
            assert precomputed_sin_theta.size == hp.nside2npix(nside), "Precomputed sin(theta) must have the same size as the number of pixels"
            sin_theta = precomputed_sin_theta
        alm_spin_raising = hp.map2alm(
            initial_map * (sin_theta ** (-input_spin)), 
            lmax=lmax
            )
        alm_spin_lowering = hp.map2alm(
            initial_map * (sin_theta ** (input_spin)), 
            lmax=lmax
            )
        _, map_dtheta_spin_raising, map_dphi_spin_raising = hp.alm2map_der1(alm_spin_raising, nside, lmax=lmax) 
        _, map_dtheta_spin_lowering, map_dphi_spin_lowering = hp.alm2map_der1(alm_spin_lowering, nside, lmax=lmax) 
        # map_dphi already contains the sin(theta) factor


    spin_output_maps = Spin_maps()

    spin_output_maps[input_spin-1] =  - (map_dtheta_spin_lowering - 1j * map_dphi_spin_lowering) * sin_theta ** (-input_spin)
    spin_output_maps[input_spin+1] =  - (map_dtheta_spin_raising + 1j * map_dphi_spin_raising) * sin_theta ** (input_spin)

    return spin_output_maps
