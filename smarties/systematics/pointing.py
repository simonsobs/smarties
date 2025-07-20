import numpy as np
import healpy as hp
from opt_einsum import contract

from smarties.hn import Spin_maps
from smarties.external.s4cmb import get_first_spin_derivative

def create_pointing_spin_leakage_map(
        intensity_CMB, 
        amplitude_offset, 
        angle_offset,
        lmax=None
    ):
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
        intensity CMB map already convolved with Gaussian circularly-symmetric beam (as assumed in the formalism), the output maps will have the same dimension
    amplitude_offset: float
        angular amplitude offset for each detector in radians
    angle_offset: float
        angle offset for each detector in radians
    
    Returns
    -------
    pointing_leakage_spin_maps: dictionary of pointing leakage maps
        dictionary of pointing leakage maps, each of shape (npix,), with keys being spin=1 and -1 

    Note
    ----
    Only the temperature leakage is considered here, the polarization leakage is not implemented
    """

    assert intensity_CMB.ndim == 1, 'The intensity_CMB map must have only 1 dimension'
    assert np.log(np.sqrt(intensity_CMB.size/12)) / np.log(2) % 1 == 0, 'The intensity_CMB map dimension must be compatible with a full sky healpy map'

    amplitude_offset = np.asarray(amplitude_offset)
    angle_offset = np.asarray(angle_offset)

    assert np.array(amplitude_offset).ndim == 1, 'The dimension of the amplitude_offset must be (n_det,)'
    assert amplitude_offset.shape == angle_offset.shape, 'The amplitude offset must have the same shape as the angle offset'

    nside = hp.npix2nside(intensity_CMB.size)
    if lmax is None:
        lmax = 2 * nside

    alms_I = hp.map2alm(intensity_CMB, lmax=lmax, iter=10)
    
    intensity_spin_derivatives = get_first_spin_derivative(
        np.vstack([alms_I, np.zeros_like(alms_I)]), 
        nside=nside,
        input_spin=0, 
    )

    # Compute the spin raising and lowering operators, knowing that the final dict must have shape {spin:np.ndarray[n_det,n_pix]}
    pointing_leakage_spin_maps = Spin_maps()

    # Spin 1
    pointing_leakage_spin_maps[1] = contract('d,p->dp', - amplitude_offset / 4 * np.exp(-1j*angle_offset), intensity_spin_derivatives[-1])

    # Spin -1
    pointing_leakage_spin_maps[-1] = contract('d,p->dp', - amplitude_offset / 4 *  np.exp(1j*angle_offset), intensity_spin_derivatives[1])


    return pointing_leakage_spin_maps
