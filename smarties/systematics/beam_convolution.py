# This file is part of SMARTIES.
# Copyright (c) 2024-2026 bers of the Simons Simons Observatory Collaboration.
# lease refer to the LICENSE file in the root of this repository.

import healpy as hp
import numpy as np

from smarties.harmonics import _alm2map_ducc0


def convert_alm_plusminus_to_spin(
    alm_plus, alm_minus, spin=2
):  # convert lbs alms from [alm_0, alm_plus, alm_minus] to [alm_0, alm_spin, alm_-spin]
    alms_pos_spin = -1 * (alm_plus + 1j * alm_minus)  # |spin| component
    alms_neg_spin = (alm_plus - 1j * alm_minus) * (-1.0) ** (
        -1 - spin
    )  # -|spin| component

    return alms_pos_spin, alms_neg_spin


def convert_alm_spin_to_plusminus(alm_pos_spin, alm_neg_spin, spin=2):

    alm_plus = -1 * (alm_pos_spin + (-1) ** (spin) * alm_neg_spin) / (2)
    alm_minus = -1 * (alm_pos_spin - (-1) ** (spin) * alm_neg_spin) / (2j)
    return alm_plus, alm_minus


def get_systematic_maps_from_alms_blms(
    alms: dict[str, np.ndarray],
    blms: dict[str, np.ndarray],
    fwhm: list[float],
    det_names: list,
    lmax: int,
    nside: int,
    mmax: int,
    pol_angles: np.ndarray,
    spins: np.ndarray | None = None,
):
    """
    Computes the systematic spin maps associeted to some blms, considering blms-blms_gauss(fwhm) to do so
    """
    n_det = len(det_names)
    if spins is None:
        spins_needed = np.arange(-mmax, mmax + 1)
    else:
        spins_needed = np.array(spins)
    spins_needed_pos = spins_needed[spins_needed >= 0]
    assert np.max(spins_needed_pos) <= mmax
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

        gaussian_blms = hp.blm_gauss(fwhm=fwhm_rad[idet], lmax=lmax, pol=True)
        gaussian_blms[1:, :] *= (
            np.exp(
                -2j * pol_angles[idet]
            )  # accounting for polarization angle of the detector
            * np.exp(
                2 * fwhm_rad[idet] ** 2 / (8 * np.log(2.0))
            )  # challinor polarisation
            * (-1 * np.sqrt(2))
        )
        gaussian_blms_full = np.zeros_like(alms_det)
        for m in range(2 + 1):
            indexes = hp.Alm.getidx(lmax, np.arange(lmax + 1), m)
            gaussian_blms_full[:, indexes] = gaussian_blms[:, indexes]

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

            output_alms = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)

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
