# This file is part of SMARTIES.
# Copyright (C) 2024-2026 Members of the Simons Observatory collaboration.
# Please refer to the LICENSE file in the root of this repository.

import logging

import healpy as hp
import numpy as np
from numba import njit, prange
from pixell import enmap, wcsutils

from smarties.hn import Spin_maps
from smarties.utils.harmonics import (
    alm2map_anypix,
    convert_alm_plusminus_to_spin,
)

logger = logging.getLogger(__name__)

__all__ = [
    "gaussian_circular_beam_blms",
    "get_beam_convolution_spins_maps",
]


def gaussian_circular_beam_blms(
    fwhm_rad: float,
    lmax: int,
    mmax: int,
    pol_angle_rad: float | None = None,
    single_precision: bool = False,
) -> np.ndarray:
    """Compute harmonic coefficients of a circular Gaussian beam.
    Uses the analytic Gaussian beam coefficients from Challinor et al. (2000,
    astro-ph/0008228), includes the polarization-angle phase factor in the blms.


    Parameters
    ----------
    fwhm_rad: float
        Full-width at half-maximum in radians.
    lmax: int
        Maximum multipole.
    mmax: int
        Maximum azimuthal index.
    pol_angle_rad: float or None (optional)
        Polarization angle in radians. If set, include polarized coefficients.
    single_precision: bool
        If True, return single-precision coefficients. Default is False (double precision).

    Returns
    -------
    np.ndarray
        Array of shape ``(ncomp, nalm)`` with ``ncomp=1`` (intensity) or ``3``
        (intensity + polarized). Component 2 equals ``1j * component_1``.

    Raises
    ------
    ValueError
        If ``mmax > lmax``.
    """
    is_polarized = pol_angle_rad is not None

    nval = hp.Alm.getsize(lmax, mmax)

    if mmax > lmax:
        raise ValueError("lmax value too small")

    if is_polarized and mmax < 2:
        raise ValueError("mmax must be 2 or more for polarized output")
    ncomp = 3 if is_polarized else 1
    dtype = np.complex64 if single_precision else np.complex128
    blms = np.zeros((ncomp, nval), dtype=dtype)
    sigmasq = fwhm_rad * fwhm_rad / (8 * np.log(2.0))

    ell_intensity = np.arange(lmax + 1)  # Only m=0 for intensity

    blms[0, hp.Alm.getidx(lmax, ell_intensity, 0)] = np.sqrt(
        (2 * ell_intensity + 1) / (4.0 * np.pi)
    ) * np.exp(-0.5 * sigmasq * ell_intensity * (ell_intensity + 1))

    if is_polarized:
        pol_angle_factor = np.exp(-2j * pol_angle_rad)
        ell_polarisation = np.arange(2, lmax + 1)  # Only m=2 for polarization

        blms[1, hp.Alm.getidx(lmax, ell_polarisation, 2)] = (
            np.sqrt((2 * ell_polarisation + 1) / (32 * np.pi))
            * np.exp(-0.5 * sigmasq * ell_polarisation * (ell_polarisation + 1))
            * pol_angle_factor
            * np.exp(2 * sigmasq)  # accounting for polarization angle of the detector
            * (-1 * np.sqrt(2))  # norm factor when going from +- 2 alms to almE almB
        )
        blms[2] = 1j * blms[1]

    return blms


# --- optimised implementation -------------------------------------------------------------

# The stage is three steps around the ducc0 synthesis: the beam convolution of the alms, the
# +/- basis conversion that produces the ducc0 input, and the assembly of the complex maps from
# the two real maps ducc0 returns. Left as numpy, each of them is single-threaded and allocates
# map-sized temporaries. The first two are fused into one kernel per (spin pair, detector) that
# reads the sky alms once and writes the ducc0 input directly, so the only large arrays alive are
# one (2, nalm) buffer and the returned maps; the assembly runs in place.


@njit(parallel=True, cache=True)
def _numba_spin_pair_to_ducc(
    alm0,
    alm_pos,
    alm_neg,
    factor0,
    factor_pos,
    factor_neg,
    row_pos,
    row_neg,
    flip,
    out_plus,
    out_minus,
    lmax,
    half,
    neg_half,
    half_unit,
):
    """Beam-convolve one spin pair and write the two ducc0 alms, one pass over the alms.

    ``alm_pos``/``alm_neg`` are the spin-+2 and spin--2 sky alms of the detector and
    ``factor_pos``/``factor_neg`` the beam factors applied to them. Rows ``row_pos`` and
    ``row_neg`` are the ones of the spin-``+s`` and spin-``-s`` output alms
    ``z_s = a_0 f_0 + 0.5 (a_+ f_+ + a_- f_-)``. ``out_plus``/``out_minus`` are the two alms
    of the E/B (``+``/``-``) basis that ducc0 synthesises: they become the real and the
    imaginary part of the spin-``+s`` map.
    """
    for m in prange(lmax + 1):
        offset = m * (2 * lmax + 1 - m) // 2
        for ell in range(m, lmax + 1):
            index = offset + ell
            a_pos = alm_pos[index]
            a_neg = alm_neg[index]
            z_pos = alm0[index] * factor0[row_pos, ell] + half * (
                a_pos * factor_pos[row_pos, ell] + a_neg * factor_neg[row_pos, ell]
            )
            z_neg = alm0[index] * factor0[row_neg, ell] + half * (
                a_pos * factor_pos[row_neg, ell] + a_neg * factor_neg[row_neg, ell]
            )
            out_plus[index] = (z_pos + flip * z_neg) * neg_half
            out_minus[index] = (z_pos - flip * z_neg) * half_unit


@njit(parallel=True, cache=True)
def _numba_spin_zero_to_ducc(
    alm0, alm_pos, alm_neg, factor0, factor_pos, factor_neg, row, out, lmax, half
):
    """Spin-0 alms of one detector: ``a_0 f_0 + 0.5 (a_+ f_+ + a_- f_-)``, one pass."""
    for m in prange(lmax + 1):
        offset = m * (2 * lmax + 1 - m) // 2
        for ell in range(m, lmax + 1):
            index = offset + ell
            out[index] = alm0[index] * factor0[row, ell] + half * (
                alm_pos[index] * factor_pos[row, ell]
                + alm_neg[index] * factor_neg[row, ell]
            )


@njit(parallel=True, cache=True)
def _numba_real_pair_to_complex(real_part, imag_part, output, sign_unit):
    """``output = real_part + sign_unit*imag_part``, written in place, pixels in parallel."""
    for index in prange(output.size):
        output[index] = real_part[index] + sign_unit * imag_part[index]


@njit(parallel=True, cache=True)
def _numba_real_to_complex(real_part, output):
    """``output = real_part`` as a complex map with a zero imaginary part, in parallel."""
    for index in prange(output.size):
        output[index] = real_part[index]


def _write_complex_pair(real_part, imag_part, output, sign_unit):
    """Write ``output = real_part + sign_unit*imag_part`` in place."""
    _numba_real_pair_to_complex(
        real_part.reshape(-1), imag_part.reshape(-1), output.reshape(-1), sign_unit
    )


def _write_real_map(real_part, output):
    """Write a real map into a complex map whose imaginary part is zero, in place."""
    _numba_real_to_complex(real_part.reshape(-1), output.reshape(-1))


def _fill_beam_factors(
    factors, beam_coeffs, spins_needed, ell_array, sqrt_factor, dtype, real_dtype
):
    """Beam factors of every spin of one detector.

    ``factors`` holds the three ``(n_spins, lmax+1)`` arrays that the alms kernels apply to the
    spin-0, spin-+2 and spin--2 sky alms; row ``s`` belongs to ``spins_needed[s]``. They are
    the beam-mode coefficients of the spin-``spins_needed[s]`` alms: the beam slice of mode
    ``m = -spins_needed[s]``, with its conjugations and ``(-1)**-s`` phase.
    """
    factor0, factor_pos, factor_neg = factors
    blm0, blmE, blmB = beam_coeffs
    curr_blm0 = np.zeros(sqrt_factor.size, dtype=dtype)
    curr_blmE = np.zeros(sqrt_factor.size, dtype=dtype)
    curr_blmB = np.zeros(sqrt_factor.size, dtype=dtype)
    lmax = sqrt_factor.size - 1

    for row, spin in enumerate(spins_needed):
        curr_blm0.fill(0)
        curr_blmE.fill(0)
        curr_blmB.fill(0)

        m_beam = -spin  # S_{spin} uses b*_{ell,-spin}
        prefactor = sqrt_factor * real_dtype((-1.0) ** int(-spin))
        idx_beam = hp.Alm.getidx(
            lmax, np.arange(abs(m_beam), lmax + 1), abs(m_beam)
        )  # Get indices for all ell where |m_beam| <= ell

        valid_lm_couple = ell_array >= abs(m_beam)  # we keep 0 when ell > |spin|

        if m_beam < 0:
            curr_blm0[valid_lm_couple] = (-1) ** (-m_beam) * np.conj(blm0[idx_beam])
            curr_blmE[valid_lm_couple] = (-1) ** (-m_beam) * np.conj(blmE[idx_beam])
            curr_blmB[valid_lm_couple] = (-1) ** (-m_beam) * np.conj(blmB[idx_beam])
        else:
            curr_blm0[valid_lm_couple] = blm0[idx_beam]
            curr_blmE[valid_lm_couple] = blmE[idx_beam]
            curr_blmB[valid_lm_couple] = blmB[idx_beam]

        curr_blm_pos2, curr_blm_neg2 = convert_alm_plusminus_to_spin(
            curr_blmE, curr_blmB, 2
        )
        factor0[row] = np.conj(curr_blm0) * prefactor
        factor_pos[row] = np.conj(curr_blm_pos2) * prefactor
        factor_neg[row] = np.conj(curr_blm_neg2) * prefactor


def _subtract_gaussian_beam(
    beam_coeffs,
    det_name,
    idet,
    fwhm,
    pol_angles_rad,
    lmax,
    mmax_beam,
    single_precision,
    n_det,
):
    """Returns the beam coefficients of one detector with a circular Gaussian beam removed."""
    blm0 = beam_coeffs[0].copy()
    blmE = beam_coeffs[1].copy()
    blmB = beam_coeffs[2].copy()

    assert pol_angles_rad is not None and len(pol_angles_rad) == n_det, (
        "You must provide polarization angles for all detectors if you want to substract the gaussian beam"
    )
    assert fwhm is not None and len(fwhm) == n_det, (
        "You must provide fwhm for all detectors if you want to substract the gaussian beam"
    )
    logger.debug(
        f"Substracting gaussian beam for detector {det_name} with fwhm {fwhm[idet]} arcmin"
    )
    gaussian_blms = gaussian_circular_beam_blms(
        fwhm_rad=np.radians(fwhm[idet] / 60),
        lmax=lmax,
        mmax=mmax_beam,
        pol_angle_rad=pol_angles_rad[idet] if pol_angles_rad is not None else None,
        single_precision=single_precision,
    )

    for m in range(min(2 + 1, mmax_beam + 1)):
        idx = hp.Alm.getidx(lmax, np.arange(m, lmax + 1), m)
        blm0[idx] -= gaussian_blms[0, idx]
        blmE[idx] -= gaussian_blms[1, idx]
        blmB[idx] -= gaussian_blms[2, idx]
    return blm0, blmE, blmB


def get_beam_convolution_spins_maps(
    alms: dict[str, np.ndarray],
    blms: dict[str, np.ndarray],
    det_names: list,
    lmax: int,
    mmax_beam: int,
    shape_pixels_output: tuple,
    fwhm: np.ndarray | None = None,
    pol_angles_rad: np.ndarray | None = None,
    spins: np.ndarray | None = None,
    wcs: wcsutils.WCS = None,
    substract_gaussian_beam=False,
    single_precision: bool = False,
    return_only_positive_spins: bool = True,
) -> Spin_maps:
    """Compute the spin maps of beam convolution
    associated to the sky and beam harmonic coefficients.

    Parameters
    ----------
    alms: dict[str, np.ndarray]
        Sky coefficients per detector (shape ``(3, nalm)``).
    blms: dict[str, np.ndarray]
        Beam coefficients per detector (shape ``(3, nalm)``).
    fwhm: list[float]
        Beam FWHM in arcmin, one per detector.
    det_names: list
        Detector identifiers (loop order).
    lmax: int
        Maximum multipole.
    mmax_beam: int
        Maximum azimuthal index.
    shape_pixels_output: tuple
        Shape of the output maps.
    pol_angles_rad: np.ndarray or None (optional)
        Polarization angles in radians, only used if ``substract_gaussian_beam`` is True.
    spins: np.ndarray or None (optional)
        Spins to compute, each together with its negative counterpart. If ``None``, use
        ``-mmax..mmax``.
    wcs: WCS object
        Geometry of the output map. If not specified, set to None and the output map will be
        computed with healpix pixelization.
    substract_gaussian_beam: bool
        Whether to substract a Gaussian beam or not, default to False.
    single_precision: bool
        If True, return single-precision coefficients. Default is False (double precision).
    return_only_positive_spins: bool
        If True (default), only the maps of the positive spins are returned.

    Returns
    -------
    Spin_maps
        Spin -> complex maps of shape ``(n_det, npix)``, either in healpix or CAR pixelization
        depending on the ``wcs`` parameter.
    """
    n_det = len(det_names)
    assert mmax_beam <= np.iinfo(np.int16).max, "mmax_beam is too large for np.int16"
    if spins is None:
        spins_needed = np.arange(-mmax_beam, mmax_beam + 1, dtype=np.int16)
    else:
        spins_needed = np.array(spins, dtype=np.int16)

    spins_needed_pos = spins_needed[spins_needed >= 0]
    spins_set = {int(spin) for spin in spins_needed}
    assert spins_set == {-spin for spin in spins_set}, (
        "Please provide every spin together with its negative counterpart, e.g. [0, 2, -2]"
    )

    assert np.max(spins_needed_pos) <= mmax_beam, (
        "The spin wanted must be smaller than mmax"
    )

    n_spins = len(spins_needed)
    spins_output = spins_needed_pos if return_only_positive_spins else spins_needed

    nalm = hp.Alm.getsize(lmax)

    assert np.all([alms[det_name].shape[1] == nalm for det_name in det_names]), (
        "All alms must have same mmax=lmax"
    )

    dtype = np.complex64 if single_precision else np.complex128
    real_dtype = np.float32 if single_precision else np.float64

    dict_spin_maps = {
        spin: np.empty((n_det,) + tuple(shape_pixels_output), dtype=dtype)
        for spin in spins_output
    }
    # mapping from spin to row in the factors arrays
    row_of_spin = {int(spin): row for row, spin in enumerate(spins_needed)}

    # Buffers reused by every detector and every spin, so that no map-sized array is allocated
    # inside the loops: the spin-±2 sky alms, the convolved alms of the current spin pair and
    # the spin-0 alms.
    sky_alm_pos = np.empty(nalm, dtype=dtype)
    sky_alm_neg = np.empty(nalm, dtype=dtype)
    convolved_alm_spin = np.empty((2, nalm), dtype=dtype)
    convolved_alm_spin_plus = convolved_alm_spin[0]
    convolved_alm_spin_minus = convolved_alm_spin[1]
    convolved_alm_spin_zero = np.empty(nalm, dtype=dtype)
    factors = tuple(np.empty((n_spins, lmax + 1), dtype=dtype) for _ in range(3))
    # Typed constants to avoid unwanted single precision upcasting to double
    half_unit = np.complex64(0.5j) if single_precision else np.complex128(0.5j)
    unit = np.complex64(1j) if single_precision else np.complex128(1j)
    neg_unit = -unit
    half = real_dtype(0.5)
    neg_half = real_dtype(-0.5)

    ell_array = np.arange(lmax + 1)
    sqrt_factor = np.sqrt(4.0 * np.pi / (2 * ell_array + 1)).astype(real_dtype)

    for idet, det_name in enumerate(det_names):
        alm0, almE, almB = alms[det_name]

        convert_alm_plusminus_to_spin(almE, almB, 2, out=(sky_alm_pos, sky_alm_neg))

        beam_coeffs = blms[det_name]

        if substract_gaussian_beam:
            beam_coeffs = _subtract_gaussian_beam(
                beam_coeffs,
                det_name,
                idet,
                fwhm,
                pol_angles_rad,
                lmax,
                mmax_beam,
                single_precision,
                n_det,
            )
        _fill_beam_factors(
            factors,
            beam_coeffs,
            spins_needed,
            ell_array,
            sqrt_factor,
            dtype,
            real_dtype,
        )

        for spin in spins_needed_pos:  # We computes the maps from the alms
            spin = int(spin)
            map_output = (
                enmap.empty((int(spin != 0) + 1,) + shape_pixels_output, wcs=wcs)
                if wcs is not None
                else None
            )
            if spin == 0:
                _numba_spin_zero_to_ducc(
                    alm0=alm0,
                    alm_pos=sky_alm_pos,
                    alm_neg=sky_alm_neg,
                    factor0=factors[0],
                    factor_pos=factors[1],
                    factor_neg=factors[2],
                    row=row_of_spin[0],
                    out=convolved_alm_spin_zero,
                    lmax=lmax,
                    half=half,
                )
                _write_real_map(
                    alm2map_anypix(
                        convolved_alm_spin_zero,
                        spin,
                        shape_pixels_output,
                        map_output,
                        lmax=lmax,
                    ),
                    dict_spin_maps[spin][idet],
                )
            else:
                _numba_spin_pair_to_ducc(
                    alm0=alm0,
                    alm_pos=sky_alm_pos,
                    alm_neg=sky_alm_neg,
                    factor0=factors[0],
                    factor_pos=factors[1],
                    factor_neg=factors[2],
                    row_pos=row_of_spin[spin],
                    row_neg=row_of_spin[-spin],
                    flip=real_dtype((-1) ** spin),
                    out_plus=convolved_alm_spin_plus,
                    out_minus=convolved_alm_spin_minus,
                    lmax=lmax,
                    half=half,
                    neg_half=neg_half,
                    half_unit=half_unit,
                )
                maps = alm2map_anypix(
                    convolved_alm_spin,
                    spin,
                    shape_pixels_output,
                    map_output,
                    lmax=lmax,
                )
                _write_complex_pair(maps[0], maps[1], dict_spin_maps[spin][idet], unit)
                if not return_only_positive_spins:
                    _write_complex_pair(
                        maps[0], maps[1], dict_spin_maps[-spin][idet], neg_unit
                    )

    return Spin_maps.from_dictionary(dict_spin_maps)
