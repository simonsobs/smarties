import numpy as np
import healpy as hp

def create_pointing_spin_leakage_map(intensity_CMB, angular_amplitude_offset, lmax=None):
    """
    Create the pointing leakage maps for a given intensity CMB map and angular amplitude offset,
    with output spins 1 and -1, resulting in maps
        $ \Tilde{S}_1 = - \frac{\rho_B}{4} \eth I$
        $ \Tilde{S}_{-1} = - \frac{\rho_B}{4} \bar{\eth} I$
    where $\eth$ and $\bar{\eth}$ are the spin raising and lowering operators and $\rho_B$ is the angular
    amplitude offset in radians.

    Parameters
    ----------
    intensity_CMB: np.ndarray
        intensity CMB map, the output maps will have the same dimension
    angular_amplitude_offset: float
        angular amplitude offset in radians
    
    Returns
    -------
    pointing_leakage_spin_maps: dictionary of pointing leakage maps
        dictionary of pointing leakage maps, each of shape (npix,), with keys being spin=1 and -1 

    Note
    ----
    Only the temperature leakage is considered here, the polarization leakage is not implemented
    """

    nside = hp.npix2nside(intensity_CMB.size)

    alm_itensity = hp.map2alm(intensity_CMB, lmax=lmax)

    _, map_I_dtheta, map_I_dphi = hp.alm2map_der1(alm_itensity, nside, lmax=lmax) 
    # map_I_dphi already contains the sin(theta) factor

    # Compute the spin raising and lowering operators
    pointing_leakage_spin_maps = dict()

    # Spin 1
    pointing_leakage_spin_maps[1] = - angular_amplitude_offset / 4 * (map_I_dtheta - 1j * map_I_dphi)

    # Spin -1
    pointing_leakage_spin_maps[-1] = - angular_amplitude_offset / 4 * (map_I_dtheta + 1j * map_I_dphi)

    return pointing_leakage_spin_maps
