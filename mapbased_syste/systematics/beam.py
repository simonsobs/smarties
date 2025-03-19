import numpy as np
import healpy as hp
from opt_einsum import contract

from mapbased_syste.hn import Spin_maps
from mapbased_syste.operators import get_naive_spin_derivative

def get_differential_ellipticity(
        intensity_CMB,
        coefficient_Taylor_expansion,
        lmax=None
    ):
    """
    Get the differential ellipticity maps for a given intensity CMB map and Taylor expansion coefficients, "
    in the formalism describe in arXiv:2011.13910, with output spins 0, 2, -2. "
    
    Parameters
    ----------
    intensity_CMB: np.ndarray
        intensity CMB map already convolved with Gaussian circularly-symmetric beam (as assumed in the formalism), the output maps will have the same dimension
    coefficient_Taylor_expansion: np.ndarray
        Taylor expansion coefficients for the differential ellipticity, the shape must be (n_det, 2), with respectively the coefficients [:,0] being multiplied to the temperature leakage in the spin 0 maps, and the coefficients [:,1] being multiplied to the temperature leakage in the spin 2 maps (and its conjugate mutliplied to the temperature leakage in the spin -2 maps)
    lmax: int
        maximum multipole for the computation of the spin derivatives of the intensity CMB map

    Returns
    -------
    differential_ellipticity_spin_maps: dictionary of differential ellipticity maps
        dictionary of differential ellipticity maps, each of shape (n_det,npix), with keys being spin=0, 2, -2 

    Notes
    -----
    The coefficients given in `coefficient_Taylor_expansion[:,0]` correspond to $\alpha_{2,xx} + \alpha_{2,yy}$
    """

    assert intensity_CMB.ndim == 1, 'The intensity_CMB map must have only 1 dimension'
    assert np.log(np.sqrt(intensity_CMB.size/12)) / np.log(2) % 1 == 0, 'The intensity_CMB map dimension must be compatible with a full sky healpy map'
    
    coefficient_Taylor_expansion = np.asarray(coefficient_Taylor_expansion)
    assert coefficient_Taylor_expansion.ndim == 2 and coefficient_Taylor_expansion.shape[1] == 2, 'The coefficient_Taylor_expansion must have 2 dimensions, [n_det, 2]'
    
    intensity_spin_1_derivatives = get_naive_spin_derivative(
        intensity_CMB, 
        0, 
        lmax=lmax
    )

    intensity_spin_2_derivatives = get_naive_spin_derivative(
        intensity_spin_1_derivatives[1],
        1,
        lmax=lmax
    )
    intensity_spin_m2_derivatives = get_naive_spin_derivative(
        intensity_spin_1_derivatives[-1],
        -1,
        lmax=lmax
    )

    differential_ellipticity_spin_maps = Spin_maps()

    # Spin 0
    differential_ellipticity_spin_maps[0] = .5 * contract('d,p->dp', coefficient_Taylor_expansion[:,0], intensity_spin_2_derivatives[0])

    # Spin 2
    differential_ellipticity_spin_maps[2] = contract('d,p->dp', coefficient_Taylor_expansion[:,1], intensity_spin_m2_derivatives[-2])

    # Spin -2
    differential_ellipticity_spin_maps[-2] = contract('d,p->dp', np.conj(coefficient_Taylor_expansion[:,1]), intensity_spin_2_derivatives[2])

    return differential_ellipticity_spin_maps
