# This file is part of SMARTIES.
# Copyright (c) 2024-2026 bers of the Simons Simons Observatory Collaboration.
# lease refer to the LICENSE file in the root of this repository.

import os
import numpy as np
from pixell import enmap, curvedsky

def filter_map_ell_cut(
        input_map, 
        mask_apodized,
        lmax_input_map,
        ell_min=None, 
        ell_max=None,
        return_alms=False
    ):
    """Filter an enmap map with an ell cut (high-pass and/or low-pass).

    Parameters
    ----------
    input_map : enmap
        Input enmap map to be filtered.
    ell_min : float, optional
        Minimum ell value for high-pass filtering. If None, no high-pass filter is applied. Default is None.
    ell_max : float, optional
        Maximum ell value for low-pass filtering. If None, no low-pass filter is applied. Default is None.
    return_alms : bool, optional
        If True, return the alms of the filtered map. Default is False.

    Returns
    -------
    filtered_map : enmap
        The filtered enmap map.
    filtered_alms : array, optional
        The alms of the filtered map, returned if return_alms is True.
    """

    alms_output = curvedsky.map2alm(
        input_map * mask_apodized, 
        lmax=lmax_input_map
    )
    
    filter_pass = np.ones(lmax_input_map+1)
    if ell_min is not None:
        lmin_filter = int(ell_min)
        filter_pass[:lmin_filter] = 0
    if ell_max is not None:
        lmax_filter = int(ell_max)
        filter_pass[lmax_filter:] = 0

    alms_output_filtered_lmin = curvedsky.almxfl(
        alms_output, 
        lfilter=filter_pass
    )

    map_output_filtered_lmin = curvedsky.alm2map(
        alms_output_filtered_lmin, 
        map=input_map.copy()
    )

    if return_alms:
        return map_output_filtered_lmin, alms_output_filtered_lmin
    return map_output_filtered_lmin

def uncouple_spectra_TT(
        input_map,
        mask_apodized, 
        lmax,
        delta_ell,
        input_map_2=None,
    ):
    """Compute the uncoupled power spectra in temperature of one enmap maps.

    Parameters
    ----------
    input_map : enmap
        First input enmap map.
    mask_apodized : enmap
        Apodized mask to be applied to the maps.
    delta_ell : int
        Width of the bins for the power spectrum.
    input_map2 : enmap
        Second input enmap map.
    
    Returns
    -------
    binned_cls : array
        Binned power spectra between the two maps.
    """

    try:
        import pymaster as nmt
    except ImportError:
        raise ImportError("pymaster is not installed. Please install it to use this function.")
    
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
        masked_on_input=False
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
            masked_on_input=False
        )
        cl_coupled = nmt.compute_coupled_cell(
            field_spin0_1, 
            field_spin0_2
        )
    else:
        field_spin0_2 = field_spin0_1
        cl_coupled = nmt.compute_coupled_cell(
            field_spin0_1,
            field_spin0_1
        )

    nmt_workspace_00_output = nmt.workspaces.NmtWorkspace(
        fl1=field_spin0_1, 
        fl2=field_spin0_2, 
        bins=binning_scheme
    )

    nmt_workspace_00_output.compute_coupling_matrix(
        field_spin0_1, 
        field_spin0_2, 
        binning_scheme
    )

    return binning_scheme.get_effective_ells(), nmt_workspace_00_output.decouple_cell(cl_coupled)

def uncouple_spectra_pol(
        input_map,
        mask_apodized, 
        lmax,
        delta_ell,
        input_map_2=None,
        purify_e=False,
        purify_b=False,
    ):

    try:
        import pymaster as nmt
    except ImportError:
        raise ImportError("pymaster is not installed. Please install it to use this function.")
    
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
        purify_b=purify_b
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
            purify_b=purify_b
        )
        cl_coupled = nmt.compute_coupled_cell(
            field_spin2_1, 
            field_spin2_2
        )
    else:
        field_spin2_2 = field_spin2_1
        cl_coupled = nmt.compute_coupled_cell(
            field_spin2_1,
            field_spin2_1
        )

    nmt_workspace_22_output = nmt.workspaces.NmtWorkspace(
        fl1=field_spin2_1, 
        fl2=field_spin2_2, 
        bins=binning_scheme
    )

    nmt_workspace_22_output.compute_coupling_matrix(
        field_spin2_1, 
        field_spin2_2, 
        binning_scheme
    )

    return binning_scheme.get_effective_ells(), nmt_workspace_22_output.decouple_cell(cl_coupled)



def uncouple_spectra_all(
        input_map,
        mask_apodized, 
        lmax,
        delta_ell,
        purify_e=False,
        purify_b=False,
    ):
    """
    Compute the decoupled power spectra in temperature and polarization of input maps.
    
    Parameters
    ----------
    input_map : enmap or np.ndarray
        Input map to be analyzed, either in enmap format (expected to be in CAR pixelization) or in numpy array format (expected to be in HEALPIX pixelization). The first dimension of the input map should have three components (T, Q, U).
    mask_apodized : enmap or np.ndarray
        Apodized mask to be applied to the input map, either in enmap format (expected to be in CAR pixelization) or in numpy array format (expected to be in HEALPIX pixelization). The mask should have the same type and the same last dimensions as the input map.
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
    ell_b : array
        Effective ell values for the binned power spectra.
    cl_coupled_00 : array
        Decoupled TT power spectrum.
    """

    try:
        import pymaster as nmt
    except ImportError:
        raise ImportError("pymaster is not installed. Please install it to use this function.")
    
    assert type(input_map) == type(mask_apodized), "The input map and the mask should have the same type (both enmaps or both healpix maps)."
    assert input_map.shape[0] == 3, "The input map should have three components (T, Q, U) as the first dimension."
    
    if type(input_map) == enmap.ndmap:
        assert input_map.shape[1:] == mask_apodized.shape, "The input map and the mask should have the same last two dimensions as two enmaps."
        wcs = input_map.wcs
    else:
        assert input_map.shape[-1] == mask_apodized.shape[-1], "The input map and the mask should have the same last dimension as two healpix maps."
        wcs = None

    

    

    field_spin0 = nmt.NmtField(
        mask=mask_apodized, 
        maps=[input_map[0]], 
        spin=0,
        wcs=wcs,
        lmax=lmax,
        lmax_mask=lmax,
        masked_on_input=False,
        purify_e=purify_e,
        purify_b=purify_b
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
        purify_b=purify_b
    )

    binning_scheme = nmt.NmtBin.from_lmax_linear(lmax, delta_ell)

    nmt_workspace_output = nmt.workspaces.NmtWorkspace(
        fl1=field_spin0, 
        fl2=field_spin2, 
        bins=binning_scheme,
        is_teb=True
    )


    nmt_workspace_output.compute_coupling_matrix(
        field_spin0, 
        field_spin2, 
        binning_scheme,
        is_teb=True
    )

    cl_coupled_00 = nmt.compute_coupled_cell(
        field_spin0,
        field_spin0
    )

    cl_coupled_02 = nmt.compute_coupled_cell(
        field_spin0, 
        field_spin2
    )

    cl_coupled_22 = nmt.compute_coupled_cell(
        field_spin2,
        field_spin2
    )

    indices_reordering = [0, 3, 6, 1, 2, 4, 5]

    return (
        binning_scheme.get_effective_ells(), 
        nmt_workspace_output.decouple_cell(
            np.vstack([cl_coupled_00, cl_coupled_02, cl_coupled_22])
        )[indices_reordering]
    )
