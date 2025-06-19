import numpy as np
import healpy as hp
import scipy.constants

boltzmann_constant = scipy.constants.k # J/K
planck_constant = scipy.constants.h  # J.s
light_speed = scipy.constants.c  # m/s

def convert_flux_mJy_to_muK(flux_mJy, frequency, beam_fwhm):
    """
    Convert flux density in mJy to temperature in K 
    using the Rayleigh-Jeans approximation.

    Parameters
    ----------
    flux_mJy: float
        Flux density in mJy.
    frequency: float
        Frequency in GHz.
    beam_fwhm: float
        Beam full width at half maximum in arcminutes.

    Returns
    -------
    temperature: float
        Temperature in muK.
    """
    
    # Convert beam FWHM from arcminutes to radians
    beam_fwhm_rad = np.radians(beam_fwhm / 60.0)
    
    # Convert flux density to temperature using the conversion formula
    temperature = (flux_mJy * 1e-3 * 1e-26) * (light_speed ** 2) / (2 * (frequency * 1e9) ** 2 * boltzmann_constant * beam_fwhm_rad ** 2)
    
    return temperature * 1e6


def get_coordinates_from_healpix_mask(mask, degree_output=True, nest=False):
    """
    Get the extremal coordinates of the pixels in a HEALPix mask,
    assuming a sqaure patch of the sky.

    Parameters
    ----------
    mask: np.ndarray
        HEALPix mask.

    Returns
    -------
    theta: np.ndarray
        Array of theta coordinates in radians.
    phi: np.ndarray
        Array of phi coordinates in radians.
    """
    
    npix = mask.shape[-1]
    nside = hp.npix2nside(npix)
    phi, theta = hp.pix2ang(nside, np.arange(npix), nest=nest, lonlat=degree_output)

    booleran_array = mask != 0

    
    return theta[booleran_array].min(), theta[booleran_array].max(), phi[booleran_array].min(), phi[booleran_array].max()

def generate_circular_profile_point_sources(
        radius_max,
        fwhm_pointsrcs,
        n_points_profile=1000
):
    """
    Generate a circular profile for point sources.

    Parameters
    ----------
    radius_max: float
        Maximum radius of the profile in arcminutes.
    fwhm_pointsrcs: float
        Full width at half maximum of the Gaussian profile in arcminutes.
    n_points_profile: int
        Number of points in the profile.

    Returns
    -------
    r: np.ndarray
        Array of distances from the center in radians.
    B: np.ndarray
        Circular Gaussian profile.
    """
    
    radius_profile = np.linspace(0, radius_max, n_points_profile)  # in arcminutes

    
    beam_point_source_profile = np.exp(-(4 * np.log(2) * radius_profile**2) / (fwhm_pointsrcs**2))
    
    return [radius_profile * 1/60 * np.pi / 180, beam_point_source_profile]
        

def generate_point_source_map(
        nside, 
        n_point_sources, 
        log_amplitude_pointsource_min=2.0,
        log_amplitude_pointsource_max=5.0,
        fwhm_pointsrcs=5.0,
        mask=None,
        return_CAR_map=False
    ):
    """
    Generate a point source map in HEALPix format.

    Parameters
    ----------
    nside: int
        HEALPix nside parameter.
    n_point_sources: int
        Number of point sources to generate.
    log_amplitude_pointsource_min: float
        Minimum logarithmic amplitude of the point sources, in base 10 (so that 10**log_amplitude_pointsource_min is in the same dimension as the output map).
    log_amplitude_pointsource_max: float
        Maximum logarithmic amplitude of the point sources, in base 10 (so that 10**log_amplitude_pointsource_max is in the same dimension as the output map).
    fwhm_pointsrcs: float
        Full width at half maximum of the point sources in arcminutes.
    mask: np.ndarray, optional
        HEALPix mask to define the area of the sky to generate the point sources in. If None, the full sky is used.
    return_CAR_map: bool, optional
        If True, return the map in CAR projection instead of HEALPix format. Default is False.
    
    Returns
    -------
    srcmap: np.ndarray
        Point source map in HEALPix format or CAR projection, depending on the value of return_CAR_map.

    """
    try:
        import pixell as pxl
        from pixell import enmap, utils, pointsrcs, reproject
    except ImportError:
        raise ImportError("pixell is not installed. Please install it to use this function.")
    
    res = np.pi/(utils.nint(np.pi/(hp.nside2resol(nside, arcmin=True) * utils.arcmin)))#/utils.arcmin
    
    if mask is None:
        theta_min, theta_max, phi_min, phi_max = -90, 90, -180, 180
        shape, wcs = enmap.fullsky_geometry(res=res, proj='car')

    else:
        theta_min, theta_max, phi_min, phi_max = get_coordinates_from_healpix_mask(mask)
        print(theta_min, theta_max, phi_min, phi_max)
        box = np.array([[theta_min, phi_min],[theta_max, phi_max]]) * pxl.utils.degree

        shape, wcs = enmap.geometry(pos=box, res=res, proj='car')


    # Generating profile for point sources
    profile_point_sources = generate_circular_profile_point_sources(
        radius_max=20*fwhm_pointsrcs,
        fwhm_pointsrcs=fwhm_pointsrcs,
        n_points_profile=100
    )


    # we choose a logspace between 100 and 10000
    amplitude_logspace = np.logspace(log_amplitude_pointsource_min, log_amplitude_pointsource_max, n_point_sources)
    # the position are random values inside omap
    dec_positions = np.random.uniform(theta_min, theta_max, n_point_sources) * np.pi / 180
    ra_positions = np.random.uniform(phi_min, phi_max, n_point_sources)  *np.pi / 180

    # we generate the sourcemap here
    srcmap = pointsrcs.sim_objects(
        shape, wcs, 
        poss=[dec_positions, ra_positions], 
        amps=amplitude_logspace, 
        profile=profile_point_sources,
        vmin=np.min(amplitude_logspace)*1e-4
    )

    if return_CAR_map:
        return enmap.enmap(srcmap, wcs=wcs)

    return reproject.map2healpix(
        enmap.enmap(srcmap, wcs=wcs), 
        nside=nside, 
        method='spline'
    )
