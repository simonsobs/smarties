# This file is part of SMARTIES.
# Copyright (c) 2024-2026 bers of the Simons Simons Observatory Collaboration.
# lease refer to the LICENSE file in the root of this repository.

import healpy as hp
import numpy as np
from pixell import enmap

from smarties.utils.harmonics import (
    alm2map_anypix,
    convert_alm_plusminus_to_spin,
    convert_alm_spin_to_plusminus,
)

__all__ = [
    "gaussian_circular_beam_blms",
    "get_beam_convolution_spins_maps"
]

def gaussian_circular_beam_alms(
    fwhm_rad: float,
    lmax: int,
    mmax: int,
    pol_angle_rad: float | None = None,
):
    """Compute harmonic coefficients of a circular Gaussian beam.
    Uses the analytic Gaussian beam coefficients from Challinor et al. (2000,
    astro-ph/0008228), includes the polarization-angle phase factor in the alms.


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
    alms = np.zeros((ncomp, nval), dtype=np.complex128)
    sigmasq = fwhm_rad * fwhm_rad / (8 * np.log(2.0))

    ell_intensity = np.arange(lmax + 1)  # Only m=0 for intensity

    alms[0, hp.Alm.getidx(lmax, ell_intensity, 0)] = np.sqrt(
        (2 * ell_intensity + 1) / (4.0 * np.pi)
    ) * np.exp(-0.5 * sigmasq * ell_intensity * (ell_intensity + 1))

    if is_polarized:
        pol_angle_factor = np.exp(-2j * pol_angle_rad)
        ell_polarisation = np.arange(2, lmax + 1)  # Only m=2 for polarization

        alms[1, hp.Alm.getidx(lmax, ell_polarisation, 2)] = (
            np.sqrt((2 * ell_polarisation + 1) / (32 * np.pi))
            * np.exp(-0.5 * sigmasq * ell_polarisation * (ell_polarisation + 1))
            * pol_angle_factor
            * np.exp(2 * sigmasq)  # accounting for polarization angle of the detector
            * (-1 * np.sqrt(2))  # norm factor when going from +- 2 alms to almE almB
        )
        alms[2] = 1j * alms[1]

    return alms


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
    wcs=None,
    substract_gaussian_beam=False,
):
    """Compute systematic spin maps from sky and beam harmonic coefficents.

    For each detector, this subtracts a symmetric Gaussian beam (computed from
    ``fwhm``) from the provided beam coefficients and then constructs the
    spin-weighted harmonic coefficients and maps for the requested spins.


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
        Maximum azimuthal index
    shape_pixels_output: tuple
        Shape of the output maps.
    pol_angles_rad: np.ndarray or None (optional)
        Polarization angles in radians, only used if ``substract_gaussian_beam`` is True
    spins: np.ndarray or None (optional)
        Spins to compute. If ``None``, use ``-mmax..mmax``.
    substract_gaussian_beam: bool
        Wether to substract a gaussian beam or not, default to False.
    Returns
    -------
    dict[int, np.ndarray]
        Spin -> complex maps of shape ``(n_det, npix)``.
    """
    n_det = len(det_names)
    if spins is None:
        spins_needed = np.arange(-mmax_beam, mmax_beam + 1)
    else:
        spins_needed = np.array(spins)
    spins_needed_pos = spins_needed[spins_needed >= 0]
    assert np.max(spins_needed_pos) <= mmax_beam, (
        "The spin wanted must be smaller than mmax"
    )
    assert np.all(spins_needed_pos) >= 0, (
        "Please provide only positive spins"
    )
    dict_spin_maps = {
        spin: np.zeros((n_det,) + shape_pixels_output, dtype=np.complex128)
        for spin in spins_needed
    }
    dict_harm_coeff = {
        spin: np.zeros((n_det, hp.Alm.getsize(lmax)), dtype=np.complex128)
        for spin in spins_needed
    }

    ell_array = np.arange(lmax + 1)
    sqrt_factor = np.sqrt(4.0 * np.pi / (2 * ell_array + 1))

    for idet, det_name in enumerate(det_names):
        alms_det = alms[det_name]
        blms_det = blms[det_name]

        alm0 = alms_det[0]
        almE = alms_det[1]
        almB = alms_det[2]
        alm_p2, alm_m2 = convert_alm_plusminus_to_spin(almE, almB, 2)

        if substract_gaussian_beam:
            blm0 = blms_det[0].copy()
            blmE = blms_det[1].copy()
            blmB = blms_det[2].copy()

            assert pol_angles_rad is not None and len(pol_angles_rad) == n_det, (
                "You must provide polarization angles for all detectors if you want to substract the gaussian beam"
            )
            assert fwhm is not None and len(fwhm) == n_det, (
                "You must provide fwhm for all detectors if you want to substract the gaussian beam"
            )
            print(
                f"Substracting gaussian beam for detector {det_name} with fwhm {fwhm[idet]} arcmin"
            )
            gaussian_blms = gaussian_circular_beam_alms(
                fwhm_rad=np.radians(fwhm[idet] / 60),
                lmax=lmax,
                mmax=mmax_beam,
                pol_angle_rad=pol_angles_rad[idet]
                if pol_angles_rad is not None
                else None,
            )

            for m in range(min(2 + 1, mmax_beam + 1)):
                idx = hp.Alm.getidx(lmax, np.arange(m, lmax + 1), m)
                blm0[idx] -= gaussian_blms[0, idx]
                blmE[idx] -= gaussian_blms[1, idx]
                blmB[idx] -= gaussian_blms[2, idx]
        else:
            blm0, blmE, blmB = blms_det[0], blms_det[1], blms_det[2]
        curr_blm0 = np.zeros(lmax + 1, dtype=np.complex128)
        curr_blmE = np.zeros(lmax + 1, dtype=np.complex128)
        curr_blmB = np.zeros(lmax + 1, dtype=np.complex128)

        for spin in spins_needed:
            curr_blm0.fill(0)
            curr_blmE.fill(0)
            curr_blmB.fill(0)

            m_beam = -spin  # Z_{spin} uses b*_{ell,-spin}

            prefactor = sqrt_factor * (-1.0) ** (-spin)
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

            curr_blm_p2, curr_blm_m2 = convert_alm_plusminus_to_spin(
                curr_blmE, curr_blmB, 2
            )

            spin_0_term = hp.almxfl(alm0, np.conj(curr_blm0) * prefactor)
            spin_plus_2_term = hp.almxfl(alm_p2, np.conj(curr_blm_p2) * prefactor)
            spin_minus_2_term = hp.almxfl(alm_m2, np.conj(curr_blm_m2) * prefactor)

            output_alms = spin_0_term + 0.5 * (spin_plus_2_term + spin_minus_2_term)

            dict_harm_coeff[spin][idet] = output_alms

    for spin in spins_needed_pos:  # We computes the maps from the alms
        for idet in range(n_det):
            map_output = (
                enmap.empty((int(spin != 0) + 1,) + shape_pixels_output, wcs=wcs)
                if wcs is not None
                else None
            )
            if spin == 0:
                dict_spin_maps[spin][idet] = alm2map_anypix(
                    dict_harm_coeff[spin][idet],
                    spin,
                    shape_pixels_output,
                    map_output,
                    lmax=lmax,
                ).reshape(shape_pixels_output)
            else:
                alm_plus, alm_minus = convert_alm_spin_to_plusminus(
                    dict_harm_coeff[spin][idet],
                    dict_harm_coeff[-spin][idet],
                    spin,
                )
                maps = alm2map_anypix(
                    np.array([alm_plus, alm_minus]),
                    spin,
                    shape_pixels_output,
                    map_output,
                    lmax=lmax,
                )
                dict_spin_maps[spin][idet] = maps[0] + 1j * maps[1]
                dict_spin_maps[-spin][idet] = (
                    maps[0] - 1j * maps[1]
                )  # negative spin maps are the complex conjugate of the positive spin maps

    return dict_spin_maps
