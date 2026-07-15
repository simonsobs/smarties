# This file is part of SMARTIES.
# Copyright (c) 2024-2026 bers of the Simons Simons Observatory Collaboration.
# lease refer to the LICENSE file in the root of this repository.

import healpy as hp
import numpy as np

from smarties.harmonics import _alm2map_ducc0


def convert_alm_plusminus_to_spin(
    alm_plus: np.ndarray, alm_minus: np.ndarray, spin: int = 2
):
    """Convert +/- basis alms coefficients to spin-weighted alms.

    Parameters
    ----------
    alm_plus: np.ndarray
        + basis coefficients.
    alm_minus: np.ndarray
        - basis coefficients (same shape as ``alm_plus``).
    spin: int (optional)
        Target spin (default 2).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(alm_pos_spin, alm_neg_spin)`` following Healpix convention
    """
    alms_pos_spin = -1 * (alm_plus + 1j * alm_minus)  # |spin| component
    alms_neg_spin = (alm_plus - 1j * alm_minus) * (-1.0) ** (
        -1 - spin
    )  # -|spin| component

    return alms_pos_spin, alms_neg_spin


def convert_alm_spin_to_plusminus(
    alm_pos_spin: np.ndarray, alm_neg_spin: np.ndarray, spin: int = 2
):
    """Convert spin-weighted alms coefficients to the +/- basis.

    Parameters
    ----------
    alm_pos_spin: np.ndarray
        +spin coefficients.
    alm_neg_spin: np.ndarray
        -spin coefficients (same shape as ``alm_pos_spin``).
    spin: int (optional)
        Spin of the input coefficients (default 2).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(alm_plus, alm_minus)`` following Healpix convention
    """

    alm_plus = -1 * (alm_pos_spin + (-1) ** (spin) * alm_neg_spin) / (2)
    alm_minus = -1 * (alm_pos_spin - (-1) ** (spin) * alm_neg_spin) / (2j)
    return alm_plus, alm_minus


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


def get_systematic_maps_from_alms_blms(
    alms: dict[str, np.ndarray],
    blms: dict[str, np.ndarray],
    fwhm: list[float],
    det_names: list,
    lmax: int,
    mmax_beam: int,
    nside: int,
    pol_angles_rad: np.ndarray,
    spins: np.ndarray | None = None,
    substract_gaussian_beam=True,
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
    nside: int
        HEALPix ``nside`` for output maps.
    pol_angles_rad: np.ndarray
        Polarization angles in radians.
    spins: np.ndarray or None (optional)
        Spins to compute. If ``None``, use ``-mmax..mmax``.
    substract_gaussian_beam: bool
        Wether to substract a gaussian beam or not, default to True.
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
    dict_spin_maps = {
        spin: np.zeros((n_det, hp.nside2npix(nside)), dtype=np.complex128)
        for spin in spins_needed
    }
    dict_harm_coeff = {
        spin: np.zeros((n_det, hp.Alm.getsize(lmax)), dtype=np.complex128)
        for spin in spins_needed
    }
    fwhm_rad = np.radians(np.array(fwhm) / 60)
    for idet, det_name in enumerate(det_names):
        alms_det = alms[det_name]
        blms_det = blms[det_name]


        # print("Gaussian blms for detector", det_name, ":", gaussian_blms.values[1])

        alm0 = alms_det[0]
        almE = alms_det[1]
        almB = alms_det[2]

        blm0 = blms_det[0].copy()
        blmE = blms_det[1].copy()
        blmB = blms_det[2].copy()
        if substract_gaussian_beam:
            gaussian_blms = gaussian_circular_beam_alms(
                fwhm_rad=fwhm_rad[idet],
                lmax=lmax,
                mmax=mmax_beam,
                pol_angle_rad=pol_angles_rad[idet],
            )

            for m in range(min(2 + 1, mmax_beam + 1)):
                idx = hp.Alm.getidx(lmax, np.arange(m, lmax + 1), m)
                blm0[idx] -= gaussian_blms[0, idx]
                blmE[idx] -= gaussian_blms[1, idx]
                blmB[idx] -= gaussian_blms[2, idx]

        for spin in spins_needed:

            m_beam = -spin  # Z_{spin} uses b*_{ell,-spin}

            ell_array = np.arange(
                0, lmax + 1
            )  # Only consider ell where |m_beam| <= ell

            prefactor = (
                np.sqrt(4.0 * np.pi / (2 * ell_array + 1)) * (-1.0) ** (-spin)
                # * pol_factor
            )

            idx_beam = hp.Alm.getidx(
                lmax, np.arange(abs(m_beam), lmax + 1), abs(m_beam)
            )  # Get indices for all ell where |m_beam| <= ell

            valid_lm_couple = ell_array >= abs(m_beam)

            curr_blm0 = np.zeros(
                lmax + 1, dtype=np.complex128
            )  # we keep 0 when ell > |spin|
            curr_blmE = np.zeros(lmax + 1, dtype=np.complex128)
            curr_blmB = np.zeros(lmax + 1, dtype=np.complex128)

            if m_beam < 0:
                curr_blm0[valid_lm_couple] = (-1) ** (-m_beam) * np.conj(blm0[idx_beam])
                curr_blmE[valid_lm_couple] = (-1) ** (-m_beam) * np.conj(blmE[idx_beam])
                curr_blmB[valid_lm_couple] = (-1) ** (-m_beam) * np.conj(blmB[idx_beam])
            else:
                curr_blm0[valid_lm_couple] = blm0[idx_beam]
                curr_blmE[valid_lm_couple] = blmE[idx_beam]
                curr_blmB[valid_lm_couple] = blmB[idx_beam]

            alm_p2, alm_m2 = convert_alm_plusminus_to_spin(almE, almB, 2)

            curr_blm_p2, curr_blm_m2 = convert_alm_plusminus_to_spin(
                curr_blmE, curr_blmB, 2
            )

            spin_0_term = hp.almxfl(alm0, np.conj(curr_blm0) * prefactor)
            spin_plus_2_term = hp.almxfl(alm_p2, np.conj(curr_blm_p2) * prefactor)
            spin_minus_2_term = hp.almxfl(alm_m2, np.conj(curr_blm_m2) * prefactor)

            output_alms = spin_0_term + 0.5 * (spin_plus_2_term + spin_minus_2_term)

            dict_harm_coeff[spin][idet] = output_alms
    for spin in spins_needed_pos:
        for idet in range(n_det):
            if spin == 0:
                dict_spin_maps[spin][idet] = _alm2map_ducc0(
                    dict_harm_coeff[spin][idet], spin, nside, lmax
                )
            else:
                alm_plus, alm_minus = convert_alm_spin_to_plusminus(
                    dict_harm_coeff[spin][idet],
                    dict_harm_coeff[-spin][idet],
                    spin,
                )
                maps =  _alm2map_ducc0(
                    np.array([alm_plus, alm_minus]), spin,nside, lmax=lmax
                )
                dict_spin_maps[spin][idet] = maps[0] + 1j * maps[1]
                dict_spin_maps[-spin][idet] = maps[0] - 1j * maps[1]

    return dict_spin_maps
