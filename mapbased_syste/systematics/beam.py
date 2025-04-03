import numpy as np
import healpy as hp
from opt_einsum import contract

from mapbased_syste.hn import Spin_maps
from mapbased_syste.external.s4cmb import get_second_spin_derivative
from mapbased_syste.tools import get_rotation_matrix

def get_ellipse_deviation(ellipticity, sigma_cs):
    """
    Get the ellipticity deviation $\Delta_\sigma$ from the input ellipticity given by:
        $$ ellpticity = (\sigma_{\rm maj}^2 - \sigma_{\rm min}^2) / (\sigma_{\rm maj}^2 + \sigma_{\rm min}^2) $$
    where $\sigma_{\rm maj}$ and $\sigma_{\rm min}$ are the major and minor axes of the ellipse, respectively, and
    defined as:
        $$ \sigma_{\rm maj} = \sigma_{\rm cs} + \Delta_\sigma / 2 $$.
        $$ \sigma_{\rm min} = \sigma_{\rm cs} - \Delta_\sigma / 2 $$.
    
    Parameters
    ----------
    ellipticity: np.ndarray
        ellipticity parameter for each detector
    sigma_cs: np.ndarray
        circularly-symmetric beam width for each detector, in arcmin

    Returns
    -------
    delta_sigma: np.ndarray
        ellipticity deviation $\Delta_\sigma$ for each detector so that the major and minor axes of the ellipse are given by:
        $$ \sigma_{\rm maj} = \sigma_{\rm cs} + \Delta_\sigma / 2 $$
        $$ \sigma_{\rm min} = \sigma_{\rm cs} - \Delta_\sigma / 2 $$
        without taking into account the rotation of the ellipse.

    """
    ellipticity = np.asarray(ellipticity)
    sigma_cs = np.asarray(sigma_cs)

    assert ellipticity.ndim == 1, 'The ellipticity map must have only 1 dimension'
    assert sigma_cs.shape == ellipticity.shape, 'The ellipticity and sigma_cs maps must have the same shape'

    

    return np.where(ellipticity != 0, 
                    2 * sigma_cs * (1 - np.sqrt(1 - ellipticity ** 2)) / ellipticity,
                    0
                    ) # the formula is not defined for ellipticity = 0, which correspond to a circular beam where the deviation is 0


def get_differential_ellipticity(
        intensity_CMB,
        ellipticity,
        ellipse_angle,
        sigma_FWHM,
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
    
    ellipticity = np.asarray(ellipticity)
    ellipse_angle = np.asarray(ellipse_angle)
    sigma_cs = np.asarray(sigma_FWHM) / ((8 * np.log(2)) ** 0.5) * np.pi/(180*60)  # convert from FWHM to sigma_cs, in radians

    assert ellipticity.ndim == 1, 'The ellipticity map must have only 1 dimension'
    assert ellipticity.shape == sigma_cs.shape, 'The ellipticity and sigma_cs maps must have the same shape'
    assert ellipticity.shape == ellipse_angle.shape, 'The ellipticity and ellipse angle maps must have the same shape'
    
    alms_I = hp.map2alm(intensity_CMB, lmax=lmax, iter=10)

    intensity_spin_2_derivatives = get_second_spin_derivative(
        np.hstack([alms_I, np.zeros(alms_I)]), 
        input_spin=0, 
        lmax=lmax
    )

    rotation_matrix_ellipse_angle = get_rotation_matrix(ellipse_angle)
    delta_sigma = get_ellipse_deviation(ellipticity, sigma_cs)

    propagation_perturbation_ellipse = np.einsum('dxy, xd, dxa->dya',
                                                 rotation_matrix_ellipse_angle,
                                                 np.vstack([delta_sigma, -delta_sigma]),
                                                 rotation_matrix_ellipse_angle
                                                )


    alpha_2 = contract('d, dxy->dxy', 
                       sigma_cs / (sigma_cs ** 2 - delta_sigma ** 2), 
                       propagation_perturbation_ellipse
                    ) - np.broadcast_to(
                        delta_sigma ** 2 / ((sigma_cs ** 2 - delta_sigma ** 2)), 
                        (sigma_cs.size, 2, 2)
                    ) * np.eye(2)

    differential_ellipticity_spin_maps = Spin_maps()

    # Spin 0
    differential_ellipticity_spin_maps[0] = contract('d,p->dp',  np.linalg.trace(alpha_2) / 2., intensity_spin_2_derivatives['+1-1'])

    spin_2_prefactor = (alpha_2[...,1,1] - alpha_2[...,0,0]) / 2. - 1j*(alpha_2[...,1,0] + alpha_2[...,0,1]) / 2.
    # Spin 2
    differential_ellipticity_spin_maps[2] = contract('d,p->dp', spin_2_prefactor, intensity_spin_2_derivatives[-2])

    # Spin -2
    differential_ellipticity_spin_maps[-2] = contract('d,p->dp', np.conj(spin_2_prefactor), intensity_spin_2_derivatives[2])

    return differential_ellipticity_spin_maps
