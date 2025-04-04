import numpy as np
import healpy as hp

from mapbased_syste.hn import Spin_maps

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
    

def create_CMB_spin_maps(nside, nstokes, lmax, fwhm=0., maps_CMB=None, seed=42):
    """
    Create spin maps for the CMB contribution, the $\Tilde{S}_k$ maps, defined
    as 
        $\Tilde{S}_0 = I$
        $\Tilde{S}_2 = \frac{1}{2} (Q + iU)$
        $\Tilde{S}_{-2} = \frac{1}{2} (Q - iU)$
    either by generating or forming them from existing CMB maps given as argument

    Parameters
    ----------
    nside: int
        nside of the maps
    nstokes: int
        number of Stokes parameters : 1 for the intensity only, 2 for the polarization only and 3 for the full Stokes parameters (T, Q, U)
    lmax: int
        maximum multipole
    fwhm: float
        full width at half maximum of the beam in arcmin ; if 0, no smoothing is applied
    maps_CMB: np.ndarray
        CMB maps ; if None, the CMB maps are generated with CAMB ; must be of shape (nstokes, npix)
    seed: int
        seed for the random generation of the CMB maps, only relevant if maps_CMB is None

    Returns
    -------
    spin_maps: dictionary of spin maps
        dictionary of spin maps, each of shape (n_spin, npix), with n_spin being 1 if nstokes=1 (spin=0), 2 if nstokes=2 (spin=2, -2) and 3 if nstokes=3 (spin=0, 2, -2) 

    """
    npix = 12*nside**2

    assert maps_CMB is None or maps_CMB.shape == (nstokes, npix), 'The shape of the CMB maps is not correct'
    assert nstokes in [1, 2, 3], 'The number of Stokes parameters must be 1 (only temperature), 2 (only polarization) or 3 (both temperature and polarization)'
    

    spin_dict_maps = Spin_maps()

    if maps_CMB is not None:
        assert maps_CMB.shape == (nstokes, npix), 'The shape of the CMB maps is not correct'
    else:
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
            
        maps_CMB = generate_CMB_map(nside, lmax, seed=seed)[relevant_indices]
        
        if fwhm != 0:
            maps_CMB = hp.smoothing(maps_CMB, fwhm=np.deg2rad(fwhm/60), lmax=lmax)

        

    if nstokes == 1:
        spin_dict_maps[0] = maps_CMB # [spin=0]
    
    if nstokes > 1:
        if nstokes == 3: 
            spin_dict_maps[0] = maps_CMB[0] # [spin=0]
        spin_dict_maps[-2] = .5*(maps_CMB[idx_polar[0]] - 1j * maps_CMB[idx_polar[1]]) # [spin=-2]
        spin_dict_maps[2] = .5*(maps_CMB[idx_polar[0]] + 1j * maps_CMB[idx_polar[1]]) # [spin=2]
        
    return spin_dict_maps
