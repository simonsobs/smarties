# This file is part of SMARTIES.
# Copyright (C) 2024 CNRS / SciPol developers
#
# SMARTIES is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SMARTIES is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SMARTIES. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import healpy as hp

from smarties.hn import Spin_maps

def generate_power_spectra_CAMB(
    nside,
    lmax=None,
    r=0,
    Alens=1,
    H0=67.5,
    ombh2=0.022,
    omch2=0.122,
    mnu=0.06,
    omk=0,
    tau=0.06,
    ns=0.965,
    As=2e-9,
    lens_potential_accuracy=1,
    nt=0,
    ntrun=0,
    type_power='total',
    typeless_bool=False,
):
    """
    Generate power spectra from CAMB
    Return [Cl^TT, Cl^EE, Cl^BB, Cl^TE]

    Parameters
    ----------
    nside: int
        nside of the maps
    lmax: int
        maximum multipole
    r: float
        tensor to scalar ratio
    Alens: float
        lensing amplitude
    H0: float
        Hubble constant
    ombh2: float
        baryon density
    omch2: float
        cold dark matter density
    mnu: float
        sum of neutrino masses
    omk: float
        curvature density
    tau: float
        optical depth
    ns: float
        scalar spectral index
    As: float
        amplitude of the primordial power spectrum
    lens_potential_accuracy: int
        lensing potential accuracy
    nt: float
        tensor spectral index
    ntrun: float
        tensor running index
    type_power: str
        type of power spectra to return
    typeless_bool: bool
        return the full power spectra if True, otherwise only the power spectrum of type type_power

    Returns
    -------
    powers: dictionary or array[float]
        dictionary of power spectra if typeless_bool is True, otherwise power spectra of type type_power
    """
    try:
        import camb
    except ImportError:
        raise ImportError('camb is not installed. Please install it with "pip install camb"')

    if lmax is None:
        lmax = 2 * nside
    # pars = camb.CAMBparams(max_l_tensor=lmax, parameterization='tensor_param_indeptilt')
    pars = camb.CAMBparams(max_l_tensor=lmax)
    pars.WantTensors = True

    pars.Accuracy.AccurateBB = True
    pars.Accuracy.AccuratePolarization = True
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau, Alens=Alens)
    pars.InitPower.set_params(As=As, ns=ns, r=r, parameterization='tensor_param_indeptilt', nt=nt, ntrun=ntrun)
    pars.max_eta_k_tensor = lmax + 100 

    # pars.set_cosmology(H0=H0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=lens_potential_accuracy)

    print('Calculating spectra from CAMB !')
    results = camb.get_results(pars)

    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True, lmax=lmax)
    if typeless_bool:
        return powers
    return powers[type_power]


def generate_CMB_map(nside, lmax, seed=42):
    """
    Returns CMB maps generated with CAMB
    """

    # Generating the CMB power spectra
    all_spectra = generate_power_spectra_CAMB(nside, typeless_bool=False).T

    # Generating the CMB map
    np.random.seed(seed)
    return hp.synfast(all_spectra, nside, lmax=lmax, new=True)
    

def create_CMB_spin_maps(nside, nstokes, lmax, fwhm=0., seed=42):
    """
    Create spin maps for the CMB contribution, the $\Tilde{S}_k$ maps, defined
    as 
        $\Tilde{S}_0 = I$
        $\Tilde{S}_2 = \frac{1}{2} (Q + iU)$
        $\Tilde{S}_{-2} = \frac{1}{2} (Q - iU)$
    by generating them from CAMB 

    Parameters
    ----------
    nside: int
        Nside of the maps
    nstokes: int
        Number of Stokes parameters : 1 for the intensity only, 2 for the polarization only and 3 for the full Stokes parameters (T, Q, U)
    lmax: int
        Maximum multipole
    fwhm: float
        Full width at half maximum of the beam in arcmin ; if 0, no smoothing is applied
    seed: int
        Seed for the random generation of the CMB maps, only relevant if maps_CMB is None

    Returns
    -------
    spin_maps: dictionary of spin maps
        dictionary of spin maps, each of shape (n_spin, npix), with n_spin being 1 if nstokes=1 (spin=0), 2 if nstokes=2 (spin=2, -2) and 3 if nstokes=3 (spin=0, 2, -2) 

    """
    npix = 12*nside**2

    assert nstokes in [1, 2, 3], 'The number of Stokes parameters must be 1 (only temperature), 2 (only polarization) or 3 (both temperature and polarization)'
    

    spin_dict_maps = Spin_maps()

    # Selecting the relevant maps
    if nstokes == 2:
        # Q, U
        relevant_indices = np.array([1, 2])
        idx_polar = np.array([0, 1])
    elif nstokes == 1:
        # I
        relevant_indices = [...] #np.array([0])
    else:
        # I, Q, U
        relevant_indices = np.arange(3)
        idx_polar = np.array([1, 2])
        
    maps_CMB = generate_CMB_map(nside, lmax, seed=seed)
    
    if fwhm != 0:
        maps_CMB = hp.smoothing(maps_CMB, fwhm=np.deg2rad(fwhm/60), lmax=lmax)
    
    maps_CMB = maps_CMB[relevant_indices]
        

    if nstokes == 1:
        spin_dict_maps[0] = maps_CMB # [spin=0]
    
    if nstokes > 1:
        if nstokes == 3: 
            spin_dict_maps[0] = maps_CMB[0] # [spin=0]
        spin_dict_maps[-2] = .5*(maps_CMB[idx_polar[0]] - 1j * maps_CMB[idx_polar[1]]) # [spin=-2]
        spin_dict_maps[2] = .5*(maps_CMB[idx_polar[0]] + 1j * maps_CMB[idx_polar[1]]) # [spin=2]
        
    return spin_dict_maps
