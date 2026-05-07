# This file is part of SMARTIES.
# Copyright (c) 2024-2026 bers of the Simons Simons Observatory Collaboration.
# lease refer to the LICENSE file in the root of this repository.

import healpy as hp
import numpy as np

from smarties.harmonics import _alm2map_ducc0


def convert_alm_plusminus_to_spin(
    alm_plus: np.ndarray, alm_minus: np.ndarray, spin: int = 2
):
    """Convert +/- basis harmonic coefficients to spin-weighted coefficients.

    Parameters
    ----------
    alm_plus, alm_minus : np.ndarray
        Complex harmonic coefficients in the +/- basis. Must have identical
        shape.
    spin : int, optional
        Target spin. Defaults to 2.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(alm_pos_spin, alm_neg_spin)``, the +spin and -spin coefficients using
        the phase convention implemented here.
    """
    alms_pos_spin = -1 * (alm_plus + 1j * alm_minus)  # |spin| component
    alms_neg_spin = (alm_plus - 1j * alm_minus) * (-1.0) ** (
        -1 - spin
    )  # -|spin| component

    return alms_pos_spin, alms_neg_spin


def convert_alm_spin_to_plusminus(
    alm_pos_spin: np.ndarray, alm_neg_spin: np.ndarray, spin: int = 2
):
    """Convert spin-weighted coefficients to the +/- basis.

    This is the inverse transform of
    :func:`convert_alm_plusminus_to_spin` for the same ``spin``.

    Parameters
    ----------
    alm_pos_spin, alm_neg_spin : np.ndarray
        Complex harmonic coefficients for +spin and -spin. Must have identical
        shape.
    spin : int, optional
        Spin of the input coefficients. Defaults to 2.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(alm_plus, alm_minus)`` in the +/- basis using the phase convention
        implemented here.
    """

    alm_plus = -1 * (alm_pos_spin + (-1) ** (spin) * alm_neg_spin) / (2)
    alm_minus = -1 * (alm_pos_spin - (-1) ** (spin) * alm_neg_spin) / (2j)
    return alm_plus, alm_minus


def gaussian_symmetric_beam_alms(
    fwhm_rad: float,
    lmax: int,
    mmax: int,
    pol_angle_rad: float | None = None,
):
    """Compute spherical harmonic coefficients of a circular Gaussian beam.

    Parameters
    ----------
    fwhm_rad : float
        Full-width at half-maximum in radians.
    lmax, mmax : int
        Maximum multipole and azimuthal index.
    pol_angle_rad : float or None, optional
        Detector polarization angle in radians. If provided, polarized beam
        coefficients are returned; otherwise only intensity is returned.

    Returns
    -------
    np.ndarray
        Complex array of shape ``(ncomp, nalm)``, where ``ncomp=1`` for
        intensity-only output and ``ncomp=3`` when polarized output is
        requested. Component 0 holds the m=0 intensity coefficients; components
        1 and 2 hold the m=2 polarized E,B coefficients following the healpix convention.

    Notes
    -----
    Uses the analytic Gaussian beam coefficients from Challinor et al. (2000,
    astro-ph/0008228), includes the polarization-angle phase factor in the alms.
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
    mmax: int,
    nside: int,
    pol_angles_rad: np.ndarray,
    spins: np.ndarray | None = None,
):
    """Compute systematic spin maps from sky and beam harmonic coefficients.

    For each detector, this subtracts a symmetric Gaussian beam (computed from
    ``fwhm``) from the provided beam coefficients and then constructs the
    spin-weighted harmonic coefficients and maps for the requested spins.

    Parameters
    ----------
    alms : dict[str, np.ndarray]
        Sky harmonic coefficients per detector. Each value is expected to be a
        complex array with shape ``(3, nalm)`` following the component ordering
        used throughout this module.
    blms : dict[str, np.ndarray]
        Beam harmonic coefficients per detector with the same shape and
        ordering as ``alms``.
    fwhm : list[float]
        Beam full-width at half-maximum values in arcminutes, one per detector.
    det_names : list
        Detector identifiers; used to index ``alms``/``blms`` and to define the
        detector loop order.
    lmax, mmax : int
        Maximum multipole and azimuthal index.
    nside : int
        HEALPix ``nside`` for the output maps.
    pol_angles_rad : np.ndarray
        Polarization angles (radians) for each detector.
    spins : np.ndarray or None, optional
        Spins to compute. If ``None``, all spins from ``-mmax`` to ``mmax`` are
        computed.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping from spin to complex maps with shape ``(n_det, npix)``. Positive
        and negative spins are both returned; spin-0 maps are real-valued but
        stored in a complex array.

    Notes
    -----
    The beam arrays in ``blms`` are modified in place when subtracting the
    symmetric Gaussian component.
    """
    n_det = len(det_names)
    if spins is None:
        spins_needed = np.arange(-mmax, mmax + 1)
    else:
        spins_needed = np.array(spins)
    spins_needed_pos = spins_needed[spins_needed >= 0]
    assert np.max(spins_needed_pos) <= mmax, "The spin wanted must be smaller than mmax"
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

        gaussian_blms = gaussian_symmetric_beam_alms(
            fwhm_rad=fwhm_rad[idet],
            lmax=lmax,
            mmax=mmax,
            pol_angle_rad=pol_angles_rad[idet],
        )

        # print("Gaussian blms for detector", det_name, ":", gaussian_blms.values[1])

        alm0 = alms_det[0]
        almE = alms_det[1]
        almB = alms_det[2]

        blm0 = blms_det[0]
        blmE = blms_det[1]
        blmB = blms_det[2]

        for m in range(min(2 + 1, mmax + 1)):
            idx = hp.Alm.getidx(lmax, np.arange(m, lmax + 1), m)
            blm0[idx] -= gaussian_blms[0, idx]
            blmE[idx] -= gaussian_blms[1, idx]
            blmB[idx] -= gaussian_blms[2, idx]

        for spin in spins_needed:
            # pol_factor = np.exp(1j * (spin) * pol_angles[idet])

            m_beam = -spin  # W_{spin} uses b*_{ell,-spin}

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
                maps = hp.alm2map_spin(
                    np.array([alm_plus, alm_minus]), nside, spin, lmax=lmax
                )
                dict_spin_maps[spin][idet] = maps[0] + 1j * maps[1]
                dict_spin_maps[-spin][idet] = maps[0] - 1j * maps[1]

    return dict_spin_maps
