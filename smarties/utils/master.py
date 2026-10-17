# This file is part of SMARTIES.
# Copyright (C) 2024-2026 Members of the Simons Observatory collaboration.
# Please refer to the LICENSE file in the root of this repository.

import os
from copy import deepcopy

import ducc0
import healpy as hp
import numpy as np
from pixell import curvedsky, enmap, wcsutils

__all__ = [
    'filter_map_ell_cut',
    'uncouple_spectra_TT',
    'uncouple_spectra_pol',
    'uncouple_cross_spectra',
    'uncouple_spectra_all',
    'uncouple_spectra_pspy'
]


def filter_map_ell_cut(
    input_map: enmap.ndmap,
    mask_apodized: enmap.ndmap,
    lmax_input_map: int,
    ell_min: int = None,
    ell_max: int = None,
    return_alms: bool = False,
) -> enmap.ndmap | tuple[enmap.ndmap, np.ndarray]:
    """
    Filter a map with an ell cut (high-pass and/or low-pass). 
    The map needs to be provided in CAR pixelization. 
    The filtering is done in harmonic space.

    Parameters
    ----------
    input_map : enmap.ndmap
        Input map to be filtered.
    mask_apodized : enmap.ndmap
        Apodized mask to be applied to the input map.
    lmax_input_map : int
        Maximum ell value for the input map.
    ell_min : int, optional
        Minimum ell value for high-pass filtering. If None, no high-pass filter is applied. Default is None.
    ell_max : int, optional
        Maximum ell value for low-pass filtering. If None, no low-pass filter is applied. Default is None.
    return_alms : bool, optional
        If True, return the alms of the filtered map. Default is False.

    Returns
    -------
    filtered_map : enmap.ndmap
        The filtered output map.
    filtered_alms : np.ndarray, optional
        The alms of the filtered map, returned if return_alms is True.
    """

    alms_output = curvedsky.map2alm(input_map * mask_apodized, lmax=lmax_input_map)

    filter_pass = np.ones(lmax_input_map + 1)
    if ell_min is not None:
        filter_pass[:ell_min] = 0
    if ell_max is not None:
        filter_pass[ell_max:] = 0

    alms_output_filtered_lmin = curvedsky.almxfl(alms_output, lfilter=filter_pass)

    map_output_filtered_lmin = curvedsky.alm2map(
        alms_output_filtered_lmin, map=input_map.copy()
    )

    if return_alms:
        return map_output_filtered_lmin, alms_output_filtered_lmin
    return map_output_filtered_lmin


def uncouple_spectra_TT(
    input_map: enmap.ndmap,
    mask_apodized: enmap.ndmap,
    lmax: int,
    delta_ell: int,
    input_map_2: enmap.ndmap = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the uncoupled power spectra in temperature of the input maps, either auto-spectra or cross-spectra
    if a second map is provided. The input maps need to be provided in CAR pixelization. 
    The computation is done using the NaMaster library.

    Parameters
    ----------
    input_map : enmap.ndmap
        First input map.
    mask_apodized : enmap.ndmap
        Apodized mask to be applied to the maps.
    lmax : int
        Maximum multipole moment to consider.
    delta_ell : int
        Width of the bins for the power spectrum.
    input_map_2 : enmap.ndmap
        Second input map (optional). If not provided, the auto-spectrum of the first map will be computed.

    Returns
    -------
    ell_b : np.ndarray
        Effective ell values for the binned power spectra.
    binned_cls : np.ndarray
        Binned decoupled power spectra.

    Notes
    -----
    The NaMaster library needs to be installed to use this function, but is not 
    installed by default with SMARTIES. Please refer to the NaMaster documentation for installation instructions.
    """

    try:
        import pymaster as nmt
    except ImportError:
        raise ImportError(
            "pymaster is not installed. Please install it to use this function."
        )

    intensity_map_1 = input_map[0] if input_map.ndim == 3 else input_map
    if input_map_2 is not None:
        intensity_map_2 = input_map_2[0] if input_map_2.ndim == 3 else input_map_2

    wcs = input_map.wcs

    field_spin0_1 = nmt.NmtField(
        mask=mask_apodized,
        maps=[intensity_map_1],
        spin=0,
        wcs=wcs,
        lmax=lmax,
        masked_on_input=False,
    )

    binning_scheme = nmt.NmtBin.from_lmax_linear(lmax, delta_ell)

    if input_map_2 is not None:
        field_spin0_2 = nmt.NmtField(
            mask=mask_apodized,
            maps=[intensity_map_2],
            spin=0,
            wcs=wcs,
            lmax=lmax,
            lmax_mask=lmax,
            masked_on_input=False,
        )
        cl_coupled = nmt.compute_coupled_cell(field_spin0_1, field_spin0_2)
    else:
        field_spin0_2 = field_spin0_1
        cl_coupled = nmt.compute_coupled_cell(field_spin0_1, field_spin0_1)

    nmt_workspace_00_output = nmt.workspaces.NmtWorkspace(
        fl1=field_spin0_1, fl2=field_spin0_2, bins=binning_scheme
    )

    nmt_workspace_00_output.compute_coupling_matrix(
        field_spin0_1, field_spin0_2, binning_scheme
    )

    return binning_scheme.get_effective_ells(), nmt_workspace_00_output.decouple_cell(
        cl_coupled
    )


def uncouple_spectra_pol(
    input_map: enmap.ndmap,
    mask_apodized: enmap.ndmap,
    lmax: int,
    delta_ell: int,
    input_map_2: enmap.ndmap = None,
    purify_e: bool = False,
    purify_b: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the decoupled power spectra in polarization of the input maps, either auto-spectra or cross-spectra
    if a second map is provided. The input maps need to be provided in CAR pixelization.

    Parameters
    ----------
    input_map : enmap.ndmap
        First input map.
    mask_apodized : enmap.ndmap
        Apodized mask to be applied to the maps.
    lmax : int
        Maximum multipole moment to consider.
    delta_ell : int
        Width of the bins for the power spectrum.
    input_map_2 : enmap.ndmap, optional
        Second input map. If not provided, the function will compute the auto-spectra of the first map.
    purify_e : bool, optional
        Whether to purify the E-mode polarization.
    purify_b : bool, optional
        Whether to purify the B-mode polarization.
    
    Returns
    -------
    ell_b : np.ndarray
        Effective ell values for the binned decoupled power spectra.
    binned_cls : np.ndarray
        Binned decoupled power spectra.

    Notes
    -----
    The NaMaster library needs to be installed to use this function, but is not 
    installed by default with SMARTIES. Please refer to the NaMaster documentation for installation instructions.
    """
    try:
        import pymaster as nmt
    except ImportError:
        raise ImportError(
            "pymaster is not installed. Please install it to use this function."
        )

    polarization_map_1 = input_map[1:] if input_map.ndim == 3 else input_map
    if input_map_2 is not None:
        polarization_map_2 = input_map_2[1:] if input_map_2.ndim == 3 else input_map_2

    wcs = polarization_map_1.wcs

    field_spin2_1 = nmt.NmtField(
        mask=mask_apodized,
        maps=polarization_map_1,
        spin=2,
        wcs=wcs,
        lmax=lmax,
        lmax_mask=lmax,
        masked_on_input=False,
        purify_e=purify_e,
        purify_b=purify_b,
    )

    binning_scheme = nmt.NmtBin.from_lmax_linear(lmax, delta_ell)

    if input_map_2 is not None:
        field_spin2_2 = nmt.NmtField(
            mask=mask_apodized,
            maps=polarization_map_2,
            spin=2,
            wcs=wcs,
            lmax=lmax,
            lmax_mask=lmax,
            masked_on_input=False,
            purify_e=purify_e,
            purify_b=purify_b,
        )
        cl_coupled = nmt.compute_coupled_cell(field_spin2_1, field_spin2_2)
    else:
        field_spin2_2 = field_spin2_1
        cl_coupled = nmt.compute_coupled_cell(field_spin2_1, field_spin2_1)

    nmt_workspace_22_output = nmt.workspaces.NmtWorkspace(
        fl1=field_spin2_1, fl2=field_spin2_2, bins=binning_scheme
    )

    nmt_workspace_22_output.compute_coupling_matrix(
        field_spin2_1, field_spin2_2, binning_scheme
    )

    return binning_scheme.get_effective_ells(), nmt_workspace_22_output.decouple_cell(
        cl_coupled
    )


def uncouple_cross_spectra(
    input_map: enmap.ndmap,
    input_map_2: enmap.ndmap,
    mask_apodized: enmap.ndmap,
    lmax: int,
    delta_ell: int,
    purify_e: bool = False,
    purify_b: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the uncoupled power spectra in temperature of one maps.

    Parameters
    ----------
    input_map : enmap.ndmap
        First input map.
    input_map_2 : enmap.ndmap
        Second input map.
    mask_apodized : enmap.ndmap
        Apodized mask to be applied to the maps.
    lmax : int
        Maximum multipole to be considered for the power spectrum estimation.
    delta_ell : int
        Width of the bins for the power spectrum.
    purify_e : bool, optional
        If True, apply E-mode purification to the polarization maps. Default is False.
    purify_b : bool, optional
        If True, apply B-mode purification to the polarization maps. Default is False.
    
    Returns
    -------
    binned_cls : np.ndarray
        Binned decoupled power spectra between the two maps.
    
    
    Notes
    -----
    The NaMaster library needs to be installed to use this function, but is not 
    installed by default with SMARTIES. Please refer to the NaMaster documentation for installation instructions.
    """

    try:
        import pymaster as nmt
    except ImportError:
        raise ImportError(
            "pymaster is not installed. Please install it to use this function."
        )

    wcs = input_map.wcs

    intensity_map_1 = input_map[0] if input_map.ndim == 3 else input_map
    if input_map_2 is not None:
        intensity_map_2 = input_map_2[0] if input_map_2.ndim == 3 else input_map_2

    polarization_map_1 = input_map[1:] if input_map.ndim == 3 else input_map
    if input_map_2 is not None:
        polarization_map_2 = input_map_2[1:] if input_map_2.ndim == 3 else input_map_2

    binning_scheme = nmt.NmtBin.from_lmax_linear(lmax, delta_ell)

    field_spin0_map1 = nmt.NmtField(
        mask=mask_apodized,
        maps=[intensity_map_1],
        spin=0,
        wcs=wcs,
        lmax=lmax,
        lmax_mask=lmax,
        masked_on_input=False,
    )

    field_spin2_map1 = nmt.NmtField(
        mask=mask_apodized,
        maps=polarization_map_1,
        spin=2,
        wcs=wcs,
        lmax=lmax,
        lmax_mask=lmax,
        masked_on_input=False,
        purify_e=purify_e,
        purify_b=purify_b,
    )

    field_spin0_map2 = nmt.NmtField(
        mask=mask_apodized,
        maps=[intensity_map_2],
        spin=0,
        wcs=wcs,
        lmax=lmax,
        lmax_mask=lmax,
        masked_on_input=False,
    )

    field_spin2_map2 = nmt.NmtField(
        mask=mask_apodized,
        maps=polarization_map_2,
        spin=2,
        wcs=wcs,
        lmax=lmax,
        lmax_mask=lmax,
        masked_on_input=False,
        purify_e=purify_e,
        purify_b=purify_b,
    )

    nmt_workspace_00_output = nmt.workspaces.NmtWorkspace(
        fl1=field_spin0_map1, fl2=field_spin0_map2, bins=binning_scheme
    )

    nmt_workspace_22_output = nmt.workspaces.NmtWorkspace(
        fl1=field_spin2_map1, fl2=field_spin2_map2, bins=binning_scheme
    )

    nmt_workspace_02_map12_output = nmt.workspaces.NmtWorkspace(
        fl1=field_spin0_map1, fl2=field_spin2_map2, bins=binning_scheme
    )

    nmt_workspace_02_map21_output = nmt.workspaces.NmtWorkspace(
        fl1=field_spin0_map2, fl2=field_spin2_map1, bins=binning_scheme
    )

    cl_coupled_00 = nmt.compute_coupled_cell(field_spin0_map1, field_spin0_map2)

    cl_coupled_22 = nmt.compute_coupled_cell(field_spin2_map1, field_spin2_map2)

    cl_coupled_02_map12 = nmt.compute_coupled_cell(field_spin0_map1, field_spin2_map2)

    cl_coupled_02_map21 = nmt.compute_coupled_cell(field_spin0_map2, field_spin2_map1)

    nmt_workspace_00_output.compute_coupling_matrix(
        field_spin0_map1, field_spin0_map2, binning_scheme
    )

    nmt_workspace_22_output.compute_coupling_matrix(
        field_spin2_map1, field_spin2_map2, binning_scheme
    )

    nmt_workspace_02_map12_output.compute_coupling_matrix(
        field_spin0_map1, field_spin2_map2, binning_scheme
    )

    nmt_workspace_02_map21_output.compute_coupling_matrix(
        field_spin0_map2, field_spin2_map1, binning_scheme
    )

    dictionary_decoupled_spectra = {
        "00": nmt_workspace_00_output.decouple_cell(cl_coupled_00),
        "22": nmt_workspace_22_output.decouple_cell(cl_coupled_22),
        "02_map12": nmt_workspace_02_map12_output.decouple_cell(cl_coupled_02_map12),
        "02_map21": nmt_workspace_02_map21_output.decouple_cell(cl_coupled_02_map21),
    }

    return binning_scheme.get_effective_ells(), dictionary_decoupled_spectra


def uncouple_spectra_all(
    input_map: np.ndarray | enmap.ndmap,
    mask_apodized: np.ndarray | enmap.ndmap,
    lmax: int,
    delta_ell: int,
    purify_e: bool = False,
    purify_b: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the decoupled power spectra in temperature and polarization of input maps.

    Parameters
    ----------
    input_map : enmap.ndmap or np.ndarray
        Input map to be analyzed, either in CAR pixelization or in HEALPIX pixelization. 
        The first dimension of the input map should have three components (T, Q, U).
    mask_apodized : enmap.ndmap or np.ndarray
        Apodized mask to be applied to the input map, either in CAR pixelization or in HEALPIX pixelization. 
        The mask should have the same type and the same last dimensions as the input map.
    lmax : int
        Maximum multipole to be considered for the power spectrum estimation.
    delta_ell : int
        Width of the bins for the power spectrum estimation.
    purify_e : bool, optional
        If True, apply E-mode purification to the polarization maps. Default is False.
    purify_b : bool, optional
        If True, apply B-mode purification to the polarization maps. Default is False.

    Returns
    -------
    ell_b : np.ndarray
        Effective ell values for the binned decoupled power spectra.
    cl_coupled : np.ndarray
        Decoupled binned power spectrum, with the ordering of the spectra being TT, EE, BB, TE, TB, EB, BE.

    Notes
    -----
    The NaMaster library needs to be installed to use this function, but is not 
    installed by default with SMARTIES. Please refer to the NaMaster documentation for installation instructions.
    """

    try:
        import pymaster as nmt
    except ImportError:
        raise ImportError(
            "pymaster is not installed. Please install it to use this function."
        )

    assert type(input_map) == type(mask_apodized), (
        "The input map and the mask should have the same type (both enmaps or both healpix maps)."
    )
    assert input_map.shape[0] == 3, (
        "The input map should have three components (T, Q, U) as the first dimension."
    )

    if type(input_map) == enmap.ndmap:
        assert input_map.shape[1:] == mask_apodized.shape, (
            "The input map and the mask should have the same last two dimensions as two enmaps."
        )
        wcs = input_map.wcs
    else:
        assert input_map.shape[-1] == mask_apodized.shape[-1], (
            "The input map and the mask should have the same last dimension as two healpix maps."
        )
        wcs = None

    field_spin0 = nmt.NmtField(
        mask=mask_apodized,
        maps=[input_map[0]],
        spin=0,
        wcs=wcs,
        lmax=lmax,
        lmax_mask=lmax,
        masked_on_input=False,
    )

    field_spin2 = nmt.NmtField(
        mask=mask_apodized,
        maps=input_map[1:],
        spin=2,
        wcs=wcs,
        lmax=lmax,
        lmax_mask=lmax,
        masked_on_input=False,
        purify_e=purify_e,
        purify_b=purify_b,
    )

    binning_scheme = nmt.NmtBin.from_lmax_linear(lmax, delta_ell)

    nmt_workspace_output = nmt.workspaces.NmtWorkspace(
        fl1=field_spin0, fl2=field_spin2, bins=binning_scheme, is_teb=True
    )

    nmt_workspace_output.compute_coupling_matrix(
        field_spin0, field_spin2, binning_scheme, is_teb=True
    )

    cl_coupled_00 = nmt.compute_coupled_cell(field_spin0, field_spin0)

    cl_coupled_02 = nmt.compute_coupled_cell(field_spin0, field_spin2)

    cl_coupled_22 = nmt.compute_coupled_cell(field_spin2, field_spin2)

    # Ordering: TT, EE, BB, TE, TB, EB, BE
    indices_reordering = [0, 3, 6, 1, 2, 4, 5]

    return (
        binning_scheme.get_effective_ells(),
        nmt_workspace_output.decouple_cell(
            np.vstack([cl_coupled_00, cl_coupled_02, cl_coupled_22])
        )[indices_reordering],
    )


def uncouple_spectra_pspy(
    input_map: enmap.ndmap,
    mask_apodized: enmap.ndmap,
    lmax: int,
    delta_ell: int,
    input_map_2: enmap.ndmap = None,
    niter: int = 0,
    spectra_ordering_output: list | None = [
        "TT",
        "TE",
        "TB",
        "ET",
        "BT",
        "EE",
        "EB",
        "BE",
        "BB",
    ],
    type_output: str = "Cl",
    return_dictionary: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the uncoupled power spectra in temperature and polarization of the input maps, either auto-spectra or cross-spectra
    if a second map is provided. The input maps need to be provided in CAR pixelization. 
    The computation is done using the pspy library.

    Parameters
    ----------
    input_map : enmap.ndmap
        First input map.
    mask_apodized : enmap.ndmap
        Apodized mask to be applied to the maps.
    lmax : int
        Maximum multipole to be considered for the power spectrum estimation.
    delta_ell : int
        Width of the bins for the power spectrum.
    input_map_2 : enmap.ndmap, optional
        Second input map.

    Returns
    -------
    binned_cls : array
        Binned power spectra between the two maps.
    
    Notes
    -----
    The pspy library needs to be installed to use this function, but is not 
    installed by default with SMARTIES. Please refer to the pspy documentation for installation instructions.
    """

    try:
        import pspy
        from pspy._mcm_fortran import mcm_compute as mcm_fortran
    except ImportError:
        raise ImportError(
            "pspy is not installed. Please install it to use this function."
        )

    if type(input_map) == enmap.ndmap:
        assert input_map.shape[1:] == mask_apodized.shape, (
            "The input map and the mask should have the same last two dimensions as two enmaps."
        )
        assert input_map.ndim == 3 and mask_apodized.ndim == 2, (
            "The input map should have three dimensions (T, Q, U) and the mask should have two dimensions for enmaps."
        )

    else:
        assert input_map.shape[-1] == mask_apodized.shape[-1], (
            "The input map and the mask should have the same last dimension as two healpix maps."
        )
        assert input_map.ndim == 2 and mask_apodized.ndim == 1, (
            "The input map should have two dimensions (3, Npix) and the mask should have one dimension for healpix maps."
        )

    boolean_different_map_2 = False
    if input_map_2 is not None:
        assert input_map_2.shape == input_map.shape, (
            "The two input maps should have the same shape."
        )
        if type(input_map) == enmap.ndmap:
            assert wcsutils.is_compatible(input_map.wcs, input_map_2.wcs), (
                "The two input enmaps should have the same WCS."
            )
        boolean_different_map_2 = True

    spectra_ordering_pspy = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

    window = pspy.so_map.from_enmap(mask_apodized)

    window.data[:] = mask_apodized

    binning_file = "binning.dat"

    n_bins = (lmax - 2) // delta_ell
    pspy.pspy_utils.create_binning_file(
        bin_size=delta_ell, n_bins=n_bins, file_name=binning_file
    )

    mbb_inv, Bbl = pspy.so_mcm.mcm_and_bbl_spin0and2(
        (window, window),
        # **dict_ell_binning,
        binning_file=binning_file,
        lmax=lmax,
        type=type_output,
        niter=niter,
    )

    alms_input_map = pspy.sph_tools.get_alms(
        pspy.so_map.from_enmap(input_map), (window, window), niter, lmax
    )
    if boolean_different_map_2:
        alm_input_map_2 = pspy.sph_tools.get_alms(
            pspy.so_map.from_enmap(input_map_2), (window, window), niter, lmax
        )
    else:
        alm_input_map_2 = alms_input_map

    ells, cl_coupled = pspy.so_spectra.get_spectra(
        alms_input_map, alm_input_map_2, spectra=spectra_ordering_pspy
    )

    output = pspy.so_spectra.bin_spectra(
        ells,
        cl_coupled,
        binning_file,
        lmax,
        type=type_output,
        mbb_inv=mbb_inv,
        spectra=spectra_ordering_pspy,
    )
    if return_dictionary:
        return output

    return output[0], np.array([output[1][spec] for spec in spectra_ordering_output])
