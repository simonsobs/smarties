# This file is part of SMARTIES.
# Copyright (C) 2024-2026 Members of the Simons Observatory collaboration.
# Please refer to the LICENSE file in the root of this repository.

import time
import logging
from copy import deepcopy
from collections import Counter
import numpy as np
from opt_einsum import contract
from numba import jit, prange
import healpy as hp
from pixell import enmap
from smarties.utils.tools import get_coupled_spin, get_row_mapmaking_matrix
from smarties.sky.cmb import create_CMB_spin_maps
from smarties.hn import Spin_maps, Spin_nm
from smarties.utils.tools import (
    _numba_spin_convolution_contract,
    _numba_apply_pixel_matrix,
    reweight_hmaps_by_hits
)
from smarties.utils.interpolation import perform_interpolation_scipy
from smarties.utils.resolution import (
    downgrade_mask,
    get_chunks_from_shape,
    get_chunks_healpix,
    get_pixels_chunk,
    get_projector_chunk,
    get_ratio_resolution,
    #TODO: BELOW TO DELETE
    get_slice_1d_from_2d,
    get_projector_map_resolution,
    get_chunks_final_map_from_downgraded_chunks,
)

logger = logging.getLogger(__name__)

__all__ = [
    'FrameworkSystematics',
    'FrameworkSystematicsChunks',
]


class FrameworkSystematics(object):
    """
    Class to simulate systematics maps, providing methods to:
    * Simulate a realization of a CMB spin maps from a given nside, lmax, possibly beam smoothed
    * Compute the inverse of the mapmaking matrix from the h_n maps
    * Compute the total maps from the h_n maps, the spin sky maps and the spin systematics maps

    The class is initialized with the map shape, the number of Stokes parameters, the maximum multipole and the list of spins involved in the signal maps.

    """

    def __init__(self, map_shape: tuple[int, ...], nstokes: int, lmax: int, list_spin_output: list[int]=[0,-2,2]):
        """
        Initialize the FrameworkSystematics class allowing to simulate systematics maps

        Parameters
        ----------
        map_shape: tuple[int]
            Total shape of the map (e.g. (n_pix,) for HEALPix maps, (ny, nx) for CAR maps),
            not reduced to the observed area
        nstokes: int
            Number of Stokes parameters : 1 for the intensity only, 2 for the polarization only and 3 for the full Stokes parameters (T, Q, U)
        lmax: int
            Maximum multipole (useful for CMB and systematics generation)
        list_spin_output: list[int]
            List of spins on which the mapmaking will be performed.
        """
        assert np.unique(list_spin_output).size == np.array(list_spin_output).size, 'The list of spins must be unique'
        assert np.all(np.isin(np.array(list_spin_output), -np.array(list_spin_output))), 'The list of spins must be symmetric around 0, i.e. if spin k is in the list, then spin -k must also be in the list'

        self.list_spin_output = list_spin_output
        self.nstokes = nstokes
        self.lmax = lmax
        self.map_shape = map_shape

    def get_CMB_spin_sky_maps(self, nside: int, fwhm: float = 0., seed: int = 42) -> dict | Spin_maps:
        """
        Get the spin CMB maps which are the following for intensity and polarization:
            * Spin 0: I
            * Spin 2: (Q + iU)/2.
            * Spin -2: (Q - iU)/2.

        Parameters
        ----------
        nside: int
            Nside of the HEALPix maps to be generated
        fwhm: float
            Full width at half maximum of the beam in arcmin ; if 0, no smoothing is applied
        seed: int
            Seed for the random generation of the CMB maps

        Returns
        -------
        spin_sky_maps: dictionary of spin sky maps (CMB, ...)
            dictionary of spin sky maps, each of shape (n_spin, npix), with n_spin being 1 if nstokes=1 (spin=0), 2 if nstokes=2 (spin=-2, 2) and 3 if nstokes=3 (spin=0, -2, 2)
        """

        return create_CMB_spin_maps(
            nside=nside, #TODO: Allow to simulate this in CAR
            nstokes=self.nstokes,
            lmax=self.lmax,
            fwhm=fwhm,
            seed=seed
        )

    def _check_h_maps_consistency(
        self, 
        h_n_spin_dict: Spin_maps,
        npix_mask: int = None,
        perform_det_interpolation: bool = False,
        precision: float = 1e-14,
    ):
        """
        Check the consistency of the h_n maps, i.e. that for each spin in the list of spins to compute, the corresponding spin h maps are provided.

        Parameters
        ----------
        h_n_spin_dict: dict | Spin_maps
            Dictionary of the summed $h$-maps, with the keys being the spins and the values the detector $h$-maps

        Returns
        -------
            Raises an assertion error if the h_n maps are not consistent
        """

        assert isinstance(h_n_spin_dict, Spin_maps), (
            'The h_n maps must be a Spin_maps object'
        )
        assert np.allclose([h_n_spin_dict[spin].ndim for spin in h_n_spin_dict.keys()], 2), 'The h_n maps must be 2D arrays of shape (n_det, n_pix)'
        if npix_mask is not None:
            assert np.all([np.allclose(h_n_spin_dict[spin].shape[-1], npix_mask) for spin in h_n_spin_dict.keys()]), f'The h_n maps must have the same number of pixels as the mask {npix_mask}'

        if perform_det_interpolation:
            assert h_n_spin_dict.shape_fullsky is not None, 'The shape of the full sky h_n maps must be provided in h_n_spin_dict.shape_fullsky for the detector interpolation to be performed'

        # Check that the h_n maps are normalized
        spin_0 = Spin_nm((0,0)) if np.array(h_n_spin_dict.spins)[0].size == 2 else 0
        assert np.max(np.abs(h_n_spin_dict[spin_0].imag) < precision), 'The h_0 maps must be real'
        assert np.max(np.abs(h_n_spin_dict[spin_0].real.sum(axis=0) - 1)) < precision, 'The h_0 maps must be normalized'

    def _check_projector_h_n(self, projector_h_n: np.ndarray, npix: int):
        """
        Check the consistency of the projector_h_n with the h_n maps, i.e. that the projector_h_n is a 1D array of integers and that its values are within the range of the number of pixels in the h_n maps.

        Parameters
        ----------
        projector_h_n: np.ndarray
            1D array of integers representing the pixel indices to project the h_n maps onto
        npix: int
            The number of pixels in the h_maps

        Returns
        -------
            Raises an assertion error if the projector_h_n is not consistent with the h_n maps
        """

        assert isinstance(projector_h_n, np.ndarray), (
            'The projector_h_n must be a numpy array'
        )
        assert projector_h_n.ndim == 1, (
            'The projector_h_n must be a 1D array'
        )
        assert np.issubdtype(projector_h_n.dtype, np.integer), (
            'The projector_h_n must be an array of integers'
        )
        assert np.all((projector_h_n >= 0) & (projector_h_n < npix)), (
            'The values in projector_h_n must be within the range of the number of pixels in the h_n maps'
        )

        assert np.all(
            np.isin(np.unique(projector_h_n), np.arange(npix))
        ), 'The projector_h_n must contain valid pixel indices'
        # assert projector_h_n.size == npix, 'The projector_h_n must have the same size as the number of pixels in the systematics maps'
        assert issubclass(
            projector_h_n.dtype.type, np.integer
        ), 'The projector_h_n must be an array of integers'


    def _check_spin_maps_sky_systematics(
        self, 
        spin_sky_maps: Spin_maps, 
        spin_systematics_maps: Spin_maps,
        dtype: np.dtype = np.complex128,
        npix: int = None
    ):
        """
        Check the consistency of the spin sky maps and the spin systematics maps,
        and add zero maps for the spins that are not provided in the input dictionaries.

        Parameters
        ----------
        spin_sky_maps: dict | Spin_maps
            Dictionary of spin sky maps, each of shape (n_spin, npix), with n_spin being 1 if nstokes=1 (spin=0), 2 if nstokes=2 (spin=-2, 2) and 3 if nstokes=3 (spin=0, -2, 2)
        spin_systematics_maps: dict | Spin_maps
            Dictionary of spin systematics maps, each of shape (n_det, npix), with n_det being the number of detectors and npix the number of pixels
        dtype: np.dtype
            Data type of the spin maps

        Returns
        -------
            Raises an assertion error if the spin sky maps and the spin systematics maps are not consistent
        """

        assert isinstance(spin_sky_maps, Spin_maps), (
            'The spin sky maps must be a Spin_maps object'
        )
        assert isinstance(spin_systematics_maps, Spin_maps), (
            'The spin systematics maps must be a Spin_maps object'
        )

        assert np.allclose([spin_sky_maps[spin].ndim for spin in spin_sky_maps.spins], 1), 'The sky maps must be 1D arrays of shape (n_pix)'
        assert np.allclose([spin_systematics_maps[spin].ndim for spin in spin_systematics_maps.spins if spin != 0 ], 2), 'The systematics maps must be 2D arrays of shape (n_det, n_pix)'

        precision = 1e-6 if np.dtype(dtype) == np.complex64 else 1e-14

        for spin in set(spin_sky_maps.spins):
            assert spin_sky_maps[spin].size > 0, (
                'The sky maps must be 1D arrays of shape (n_pix)'
            )
            if npix is not None:
                assert spin_sky_maps[spin].size == npix, (
                    f'The sky maps of spin {spin} must be 1D arrays of shape (n_pix), where n_pix={npix}, but are size {spin_sky_maps[spin].size}'
                )

            if ((spin in spin_sky_maps) and (-spin in spin_sky_maps)) and (spin != 0):
                assert np.max(np.abs(
                    spin_sky_maps[spin].imag+spin_sky_maps[-spin].imag
                )) < precision, 'The sum of the input sky maps must be real, the total imaginary part is not expected to be non-zero'
            elif spin in spin_sky_maps and spin == 0:
                assert np.max(np.abs(spin_sky_maps[spin].imag)) < precision, 'The input sky maps must be real, the imaginary part is not expected to be non-zero'
        
        for spin in set(spin_systematics_maps.spins):
            assert spin_systematics_maps[spin].size > 0, (
                'The systematics maps must be 2D arrays of shape (n_det, n_pix)'
            )
            if npix is not None:
                assert spin_systematics_maps[spin].shape[-1] == npix, (
                    f'The systematics maps of spin {spin} must be 2D arrays of shape (n_det, n_pix), where n_pix={npix} but are shape {spin_systematics_maps[spin].shape}'
                )

            if ((spin in spin_systematics_maps) and (-spin in spin_systematics_maps)) and (spin != 0):
                assert np.max(np.abs(
                    spin_systematics_maps[spin].imag+spin_systematics_maps[-spin].imag
                )) < precision, 'The sum of the input systematics maps must be real, the total imaginary part is not expected to be non-zero'
            elif spin in spin_systematics_maps and spin == 0:
                assert np.max(np.abs(spin_systematics_maps[spin].imag)) < precision, 'The input systematics maps must be real, the imaginary part is not expected to be non-zero'


        
    def _check_consistency_spin_maps(
        self, 
        set_spin_maps: set,
        h_n_spin_dict: Spin_maps
    ):
        """
        Check the consistency of the spin h maps, i.e. that for each spin in the list of spins to compute, the corresponding spin h maps are provided.
        """
        spin_h_maps = sorted(set(h_n_spin_dict.spins) | {-s for s in h_n_spin_dict.spins if s > 0})
        for spin_sky in set_spin_maps:
            for spin_output in self.list_spin_output:
                assert (spin_output-spin_sky) in spin_h_maps or -(spin_output-spin_sky) in spin_h_maps, f'The spin {spin_output-spin_sky} or {-spin_output+spin_sky} is not provided in the h_n maps, but is required for the mapmaking process'
                assert h_n_spin_dict[spin_output-spin_sky].ndim == 2 or h_n_spin_dict[-spin_output+spin_sky].ndim == 2, f'The spin {spin_output-spin_sky} or {-spin_output+spin_sky} should be a 2D array of shape (n_det, n_pix), but is not'


    def get_inverse_mapmaking_matrix(
            self,
            h_n_spin_dict: dict | Spin_maps,
            npix: int = None,
            polar_angle_coeff: np.ndarray = None,
            polar_efficiency_coeff: np.ndarray = None,
            slice_to_apply: slice = None
        ) -> np.ndarray:
        """
        Compute the inverse of the mapmaking matrix from the h_n maps

        Parameters
        ----------
        h_n_spin_dict: dict or Spin_maps
            Dictionary of the summed $h$-maps, with the keys being the spins and the values the detector $h$-maps
        npix: int
            Number of pixels in the observed area, default is None, then the number of pixels is inferred from the input maps
        polar_angle_coeff: np.ndarray
            The polar angle coefficients to use for the mapmaking, if None, they will be set to 1.
        polar_efficiency_coeff: np.ndarray
            The polar efficiency coefficients to use for the mapmaking, if None, they will be set to 1.
        slice_to_apply: slice
            Slice to apply to the h_n maps, to only compute the mapmaking matrix for a portion of the pixels.
            If None, no slice is applied. Default is None.

        Returns
        -------
        inverse_mapmaking_matrix: np.ndarray
            The inverse of the mapmaking matrix, with the shape (npix, nstokes, nstokes), with npix being the number of pixels in the observed area of the provided mask

        Note
        ----
        This function assumes that all the necessary spins are provided in the h_n maps
        and that the h_n maps are normalized
        """

        list_spin = np.array(list(h_n_spin_dict.keys()))
        dtype = h_n_spin_dict[list_spin[list_spin != 0][0]].dtype

        if slice_to_apply is None:
            slice_to_apply = ...
        elif np.all(... not in np.array(slice_to_apply)):
            slice_to_apply = ..., slice_to_apply

        if npix is None:
            npix = h_n_spin_dict[list_spin[list_spin != 0][0]][slice_to_apply].shape[-1]

        # First, form the mapmaking matrix composed of the h_n map
        mapmaking_matrix = np.zeros(
            (npix, self.nstokes, self.nstokes),
            dtype=dtype
        )
        for i, reference_spin in enumerate(self.list_spin_output):
            mapmaking_matrix[:,:,i] = get_row_mapmaking_matrix(
                reference_spin=reference_spin,
                h_n_spin_dict=h_n_spin_dict,
                list_spin_input=self.list_spin_output,
                dtype=dtype,
                polar_angle_coeff=polar_angle_coeff,
                polar_efficiency_coeff=polar_efficiency_coeff,
                slice_to_apply=slice_to_apply
            )
        tol = 1e-6 if dtype == np.complex64 else 1e-14
        hermitian = np.allclose(mapmaking_matrix, mapmaking_matrix.conj().transpose((0, 2, 1)), atol=tol, rtol=tol)
        if not hermitian:
            logger.warning("WARNING: The mapmaking matrix is not Hermitian, the building of the mapmaking matrix may be incorrect, check the h_n maps and the polar angle coefficients")
        # Then, compute the inverse of the mapmaking matrix

        return np.linalg.pinv(mapmaking_matrix, hermitian=hermitian)


    def compute_spin_convolution(
        self,
        spin_sky_maps: dict | Spin_maps,
        spin_systematics_maps: dict | Spin_maps,
        h_n_spin_dict: dict | Spin_maps,
        npix: int,
        polar_angle_coeff: np.ndarray,
        polar_efficiency_coeff: np.ndarray,
        dtype: np.dtype = np.complex128,
        projector_h_n: np.ndarray = None,
        return_array: bool = False,
    ) -> dict | np.ndarray:
        """
        Compute the spin convolution of the h_n maps with the spin sky maps and the spin systematics maps.

        Parameters
        ----------
        spin_sky_maps: dict | Spin_maps
            Dictionary (Spin_maps object) of spin sky maps
        spin_systematics_maps: dict | Spin_maps
            Dictionary (Spin_maps object) of spin systematics maps
        h_n_spin_dict: dict | Spin_maps
            Dictionary (Spin_maps object) of h_n maps
        npix: int
            Number of pixels in the maps
        polar_angle_coeff: np.ndarray
            The polar angle coefficients to use for the mapmaking, if None, they will be set to 1.
        polar_efficiency_coeff: np.ndarray
            The polar efficiency coefficients to use for the mapmaking, if None, they will be set to 1.
        dtype: np.dtype = np.complex128
            The data type to use for the output arrays.
        projector_h_n: np.ndarray (optional)
            The projector to apply to the h_n maps, if None, no projector is applied: column of the h_n maps
            (index along their pixel axis) of each of the npix pixels. The numba computation reads the h_n maps
            through these indices, without copying them.
        return_array: bool = False
            If True, return a single array with all the coupled maps. If False, return a dictionary with one array per spin.

        Returns
        -------
        dict | np.ndarray
            The coupled maps as a dictionary or a single array.
        """

        base_spins = list(np.unique(spin_sky_maps.spins + spin_systematics_maps.spins))
        list_spin_maps = sorted(set(base_spins) | {-s for s in base_spins if s > 0})


        def polar_efficiency_func(spin):
            if spin in [-2, 2]:
                return polar_efficiency_coeff
            else:
                return np.ones_like(polar_efficiency_coeff, dtype=polar_efficiency_coeff.dtype)

        # The h-maps of the pixel i are the column projector_h_n[i] of the h-maps (column i if
        # None), and those of a missing spin -n the conjugate of the spin n: the numba kernel
        # reads them through their indices without copying them, numpy copies the columns read
        if projector_h_n is None or projector_h_n is Ellipsis:
            projector_h_n, columns_h_n = np.arange(npix), ... # Use all pixels
        else:
            projector_h_n = np.asarray(projector_h_n)
            columns_h_n = (..., projector_h_n)
 

        factor_func = lambda spin: 1 if np.sum(spin) == 0 else .5


        logger.info("Optimizing the computations with numba for the spin convolution...")
        # Precompute the coupled spins for each spin in the list of spins to compute

        dict_coupled_maps = Spin_maps.from_dictionary(dict())
        for i, spin in enumerate(self.list_spin_output):
                if spin >= 0:
                    dict_coupled_maps[spin] = np.zeros((npix,), dtype=dtype)

        spin_h_maps = sorted(set(h_n_spin_dict.spins) | {-s for s in h_n_spin_dict.spins if s > 0})
        list_spin_output_positive = [spin for spin in self.list_spin_output if spin >= 0]
        
        for i, spin in enumerate(list_spin_output_positive):
            # Get all combinations of spins (k-k', k') such that k-k' = spin
            coupled_spins = get_coupled_spin(
                reference_spin=spin,
                available_h_n_spin=spin_h_maps,
                available_signal_spins=list_spin_maps
            )

            logger.info(f'Coupled spins for spin {spin}: {coupled_spins}')

            factor_correction = lambda spin: 1
            if spin == 0:
                coupled_spins = [tuple_spin for tuple_spin in coupled_spins if tuple_spin[0] >= 0]
                factor_correction = lambda spin: 1 if spin != 0 else .5
            

            dtype_scalar = np.dtype(dtype).type # np.complex64 or np.complex128 depending on single or double precision
            zero = dtype_scalar(0)

            # \sum_{k' = -\infty}^{\infty} h_{k-k'} S_{k'} on all (k-k', k') pairs
            for tuple_spins in coupled_spins:
                
                conjugate_h_maps = tuple_spins[0] not in h_n_spin_dict
                spin_to_pass_hmaps = -tuple_spins[0] if conjugate_h_maps else tuple_spins[0]

                conjugate_systematics_maps = tuple_spins[1] not in spin_systematics_maps
                if tuple_spins[1] in spin_systematics_maps or -tuple_spins[1] in spin_systematics_maps:
                    spin_to_pass_systematics = (
                        -tuple_spins[1] if conjugate_systematics_maps else tuple_spins[1]
                    )
                    systematics_maps_to_use = spin_systematics_maps[spin_to_pass_systematics]
                else:
                    systematics_maps_to_use = None

                conjugate_sky_maps = tuple_spins[1] not in spin_sky_maps
                if tuple_spins[1] in spin_sky_maps or -tuple_spins[1] in spin_sky_maps:
                    spin_to_pass_sky = -tuple_spins[1] if conjugate_sky_maps else tuple_spins[1]
                    sky_maps_to_use = spin_sky_maps[spin_to_pass_sky]
                else:
                    sky_maps_to_use = None

                if systematics_maps_to_use is None and sky_maps_to_use is None:
                    continue  # no signal of spin k'

                _numba_spin_convolution_contract(
                    h_maps = h_n_spin_dict[spin_to_pass_hmaps],
                    systematics_maps = systematics_maps_to_use,
                    sky_maps = sky_maps_to_use,
                    polar_efficiency_coeff = polar_efficiency_func(spin),
                    polar_angle_coeff_sky = polar_angle_coeff[-tuple_spins[1]],
                    polar_angle_coeff_mapmaker = polar_angle_coeff[spin],
                    prefactor = dtype_scalar(factor_func(spin)*factor_correction(tuple_spins[0])),
                    zero = zero,
                    output = dict_coupled_maps[spin],
                    projector_h_n = projector_h_n,
                    conjugate_h_maps = conjugate_h_maps,
                    conjugate_systematics_maps = conjugate_systematics_maps,
                    conjugate_sky_maps = conjugate_sky_maps
                )
            # if spin == 0:
            #     dict_coupled_maps[-spin] += np.conj(dict_coupled_maps[spin])
            if spin == 0:
                dict_coupled_maps[spin].real *= 2
                dict_coupled_maps[spin].imag = 0

        if return_array:
            spin_coupled_maps = np.zeros((npix, len(self.list_spin_output),), dtype=dtype)
            for i, spin in enumerate(self.list_spin_output):
                spin_coupled_maps[:, i] = dict_coupled_maps[spin]
            return spin_coupled_maps 
        return dict_coupled_maps


    def compute_total_maps(
            self,
            h_n_spin_dict: dict | Spin_maps,
            spin_sky_maps: dict | Spin_maps,
            spin_systematics_maps: dict | Spin_maps = None,
            mask: np.ndarray | enmap.ndmap = None,
            inverse_mapmaking_matrix : np.ndarray = None,
            return_Q_U: bool = False,
            return_inverse_mapmaking_matrix: bool = False,
            polar_angle: np.ndarray = None,
            polar_efficiency_coeff: np.ndarray = None,
            projector_h_n: np.ndarray = None,
            geometry_dictionary: dict = dict(),
            single_precision: bool = False,
        ) -> np.ndarray | Spin_maps | tuple[np.ndarray | Spin_maps, np.ndarray]:
        """
        Compute the total estimated maps from the $h$-maps, the spin sky maps and the spin systematics maps

        Parameters
        ----------
        h_n_spin_dict: dict or Spin_maps
            Dictionary of the summed $h$-maps, with the keys being the spins and the values the $h$-maps
        spin_sky_maps: dict or Spin_maps
            Dictionary of the spin sky maps (detector-independent), with the keys being the spins and the values the spin sly maps
            (e.g. if nstokes=3, the keys are 0, -2, 2 and the fields (I, (Q-iU)/2., (Q+iU)/2))
        spin_systematics_maps: dict or Spin_maps (optional)
            Dictionary of the spin systematics maps (detector-dependent), with the keys being the spins and the values the spin systematics maps,
            of shape (n_det, npix), with n_det being the number of detectors and npix the number of pixels.
            If None, no systematics are considered (default is None).
        mask: np.ndarray
            Mask of the maps, only the pixels in the observed area will be considered for the inversion
        inverse_mapmaking_matrix: np.ndarray (optional)
            The inverse of the mapmaking matrix to use, with the shape (npix, nstokes, nstokes), with npix being the number of pixels
            in the observed area of the provided mask. If None, the inverse mapmaking matrix will be computed from the provided h_n maps.
            Default is None.
        return_Q_U: bool (optional)
            If True, return the Q and U maps instead of the spin -2 and 2 maps, default is False.
        return_inverse_mapmaking_matrix: bool
            If True, return the inverse of the mapmaking matrix, default is False.
        polar_angle: np.ndarray (optional)
            The detector polarization angle to use for the mapmaking, if None, the polarization angle will be set to 0.
            Must be provided in radians. Default is None.
        polar_efficiency_coeff: np.ndarray (optional)
            The detector polarization efficiency coefficients to use for the mapmaking, if None, they will be set to 1. Default is None.
        projector_h_n: np.ndarray (optional)
            The projector to apply to the h-maps (applied as indices), if None, no projector is applied. Default is None.
        geometry_dictionary: dict (optional)
            A dictionary containing the geometry information for the mapmaking process. Must contain 'wcs_car' if the maps are in CAR format.
            Default is an empty dictionary.
        single_precision: bool (optional)
            If True, use single precision for the computations, otherwise use double precision. Default is False.


        Returns
        -------
        final_sky_fields: np.ndarray
            The final sky fields, with the shape (npix, nstokes) if return_Q_U is False, (npix, 3) otherwise
        """

        dtype = np.complex64 if single_precision else np.complex128
        precision = 1e-6 if single_precision else 1e-14

        if mask is None:
            mask = np.ones(self.map_shape, dtype=np.int8)

        npix = np.count_nonzero(mask)
        assert npix > 0, 'The mask must contain at least one non-zero pixel'

        self._check_h_maps_consistency(h_n_spin_dict, precision=precision, npix_mask=npix)

        if projector_h_n is not None:
            self._check_projector_h_n(projector_h_n, npix=npix)
        else:
            projector_h_n = ... # Use all pixels

        if spin_systematics_maps is None:
            logger.warning("No systematics maps provided, assuming no systematics")
            spin_systematics_maps = Spin_maps.from_dictionary({})

        spin_h_maps = sorted(set(h_n_spin_dict.spins) | {-s for s in h_n_spin_dict.spins if s > 0})

        # Check the consistency of the spin sky maps and the spin systematics maps
        self._check_spin_maps_sky_systematics(
            spin_sky_maps = spin_sky_maps, 
            spin_systematics_maps = spin_systematics_maps, 
            dtype = dtype,
            npix = npix
        )

        set_spins_maps = set(np.unique(spin_sky_maps.spins + spin_systematics_maps.spins))
        self._check_consistency_spin_maps(
            set_spin_maps = set_spins_maps,
            h_n_spin_dict = h_n_spin_dict,
        )
        all_spins = sorted(set_spins_maps | set(spin_h_maps) | set(self.list_spin_output))

        
        spin_0 = Spin_nm((0,0)) if np.array(h_n_spin_dict.spins)[0].size == 2 else 0
        n_det = h_n_spin_dict[spin_0].shape[0]
        if polar_angle is None:
            polar_angle_coeff = {spin: np.ones(n_det, dtype=dtype) for spin in all_spins} 
            # Default is to not apply any polar angle, i.e. the detectors are all aligned in the same direction
        else:
            assert polar_angle.size == n_det, 'The polar angle map must have the same shape as the h-maps'

            polar_angle_coeff = {spin: np.exp(spin * 1j * polar_angle).astype(dtype) for spin in all_spins}
            #TODO: Generalize to m != 0 for HWP angles

        if polar_efficiency_coeff is not None:
            assert polar_efficiency_coeff.size == n_det, 'The polar efficiency map must have the same shape as the h-maps'
        else:
            polar_efficiency_coeff = np.ones(n_det)

        if inverse_mapmaking_matrix is None:
            logger.info("Starting the computation of the inverse mapmaking matrix...")
            time_inverse_mapmaking_matrix = time.time()
            inverse_mapmaking_matrix = self.get_inverse_mapmaking_matrix(
                h_n_spin_dict,
                polar_angle_coeff=polar_angle_coeff,
                polar_efficiency_coeff=polar_efficiency_coeff,
                npix=npix,
            )
            logger.info(f"Inverse mapmaking matrix computed in {time.time() - time_inverse_mapmaking_matrix:.2f} seconds")
        else:
            assert inverse_mapmaking_matrix.shape == (npix, self.nstokes, self.nstokes), 'The inverse mapmaking matrix must be of shape (npix, nstokes, nstokes), with npix being the number of pixels in the observed area of the provided mask'

        logger.info("Computing the spin coupled maps...")
        # Second, form the data vector composed of (<d_j>, <d_j cos 2\phi_j>, <d_j sin 2\phi_j>)

        time_spin_convolution = time.time()
        spin_coupled_maps = self.compute_spin_convolution(
            spin_sky_maps = spin_sky_maps,
            spin_systematics_maps = spin_systematics_maps,
            h_n_spin_dict = h_n_spin_dict,
            projector_h_n = projector_h_n,
            npix = npix,
            polar_angle_coeff=polar_angle_coeff,
            polar_efficiency_coeff=polar_efficiency_coeff,
            dtype = dtype,
            return_array = True,
        )
        logger.info(f"Spin coupled maps computed in {time.time() - time_spin_convolution:.2f} seconds, computing the final maps ...")

        wcs_car = geometry_dictionary.get('wcs_car', None)
        get_final_map = lambda x: x if wcs_car is None else enmap.ndmap(
            x,
            wcs=wcs_car
        )

        time_final_maps = time.time()
        
        dict_final_sky_fields = Spin_maps.from_dictionary(
            {spin: np.zeros((npix,), dtype=dtype) for spin in self.list_spin_output if spin >= 0}
        )
        for i, spin in enumerate(self.list_spin_output):
            if spin<0:
                continue
            
            _numba_apply_pixel_matrix(
                matrix_line=inverse_mapmaking_matrix[projector_h_n,i,:],
                vector=spin_coupled_maps,
                output=dict_final_sky_fields[spin]
            )

        logger.info(f"Final maps computed in {time.time() - time_final_maps:.2f} seconds...")

        if return_Q_U:
            final_Q = (dict_final_sky_fields[-2] + dict_final_sky_fields[2])/2.
            final_U = 1j*(dict_final_sky_fields[-2] - dict_final_sky_fields[2])/2.
            if self.nstokes == 3:
                final_I = dict_final_sky_fields[0]
                output = np.vstack([final_I, final_Q, final_U])
            else:
                output = np.vstack([final_Q, final_U])
        else:
            output = dict_final_sky_fields

        if return_inverse_mapmaking_matrix:
            return output, inverse_mapmaking_matrix

        return output

class FrameworkSystematicsChunks(FrameworkSystematics):
    """
    :class:`smarties.mapmaking.FrameworkSystematics` with a mapmaking per chunk working on
    CAR and HEALPix maps, with $h$-maps interpolated over the focal plane and upgraded to the
    output resolution on the fly.
    """

    def _check_interpolation(self, dict_utils_interpolation_det):
        assert dict_utils_interpolation_det is not None, (
                'The utils for the interpolation of the detector maps must be provided in the parameters with key "dict_utils_interpolation_det"'
            )
        assert 'dict_positions_detector_known' in dict_utils_interpolation_det, 'The positions of the known detectors must be provided in the utils for the interpolation of the detector maps with key "dict_positions_detector_known"'
        assert 'dict_positions_detector_unknown' in dict_utils_interpolation_det, 'The positions of the unknown detectors must be provided in the utils for the interpolation of the detector maps with key "dict_positions_detector_unknown"'
        assert 'dict_key_positions' in dict_utils_interpolation_det, 'The keys for the positions of the detectors must be provided in the utils for the interpolation of the detector maps with key "dict_key_positions"'
        
        dict_positions_detector_unknown = dict_utils_interpolation_det['dict_positions_detector_unknown']
        dict_positions_detector_known = dict_utils_interpolation_det['dict_positions_detector_known']

        wafer_known = dict_utils_interpolation_det.get('wafer_list_detector_known', None)
        wafer_unknown = dict_utils_interpolation_det.get('wafer_list_detector_unknown', None)

        assert (wafer_known is None) == (wafer_unknown is None), 'The wafer lists of the known and unknown detectors must be provided together in the utils for the interpolation of the detector maps with keys "wafer_list_detector_known" and "wafer_list_detector_unknown"'

        if wafer_known is not None:
            counter_known = Counter(wafer_known)
            counter_unknown = Counter(wafer_unknown)

            for key, count in counter_known.items():
                assert count >= 3, f'The wafer {key} of the known detectors must have at least 3 detectors to perform a linear interpolation, but it has only {count} detectors'
            for key, count in counter_unknown.items():
                assert key in wafer_known, f'The wafer {key} of the unknown detectors must be present in the wafer list of the known detectors to perform an interpolation, but it is not present' 

            logger.info(f'Interpolation of the detector maps will be performed separately for each wafer, with {len(counter_known)} wafers of the known detectors and {len(counter_unknown)} wafers of the unknown detectors')
            logger.info(f'Wafer list and corresponding number of detectors of the known detectors: {counter_known}')
            logger.info(f'Wafer list and corresponding number of detectors of the unknown detectors: {counter_unknown}')

        assert 'reweight_by_hits' in dict_utils_interpolation_det, 'The boolean "reweight_by_hits" must be provided in the utils for the interpolation of the detector maps with key "reweight_by_hits"'

        if not dict_utils_interpolation_det['reweight_by_hits']:
            logger.warning("'reweight_by_hits' is False: the h-maps of the known detectors are interpolated as hits and averages, not as h-maps")

        position_x_key = dict_utils_interpolation_det['dict_key_positions']['position_x_key']
        return np.size(dict_positions_detector_unknown[position_x_key]) + np.size(dict_positions_detector_known[position_x_key])

    def _check_template_systematics_maps(
        self, 
        dict_utils_template_systematics_maps,
        dtype: np.dtype = np.complex128
    ):
        
        assert dict_utils_template_systematics_maps is not None, 'The template systematics maps must be provided with dict_utils_template_systematics_maps'
        assert 'template_spin_systematics_maps' in dict_utils_template_systematics_maps, 'The template systematics maps must be provided in the parameters template systematics maps with key "template_spin_systematics_maps"'
        assert 'parameters_template_systematics' in dict_utils_template_systematics_maps, 'The parameters of the template systematics maps must be provided in the parameters template systematics maps with key "parameters_template_systematics"'

        parameters_template_systematics = dict_utils_template_systematics_maps['parameters_template_systematics']
        template_spin_systematics_maps = dict_utils_template_systematics_maps['template_spin_systematics_maps']

        assert isinstance(template_spin_systematics_maps, Spin_maps), 'The template systematics maps must be a Spin_maps instance'
        assert isinstance(parameters_template_systematics, dict), 'The parameters of the template systematics maps must be provided as a dictionary with the keys being the spins and the values the parameters of the template systematics maps'
        assert np.all([
            spin in parameters_template_systematics
            or -spin in parameters_template_systematics
            for spin in template_spin_systematics_maps.keys()
        ]), 'The parameters of the template systematics maps must be provided for all the spins present in the template systematics maps'


        precision = 1e-6 if dtype == np.complex64 else 1e-14
        for spin in template_spin_systematics_maps.keys():
            if spin < 0:
                assert -spin in template_spin_systematics_maps, f'The template systematics maps must be provided for both positive spins at least: {-spin}.'

            if -spin in template_spin_systematics_maps:
                assert np.max(np.abs(
                    template_spin_systematics_maps[spin].imag +
                    template_spin_systematics_maps[-spin].imag)) < precision, (
                f'The template systematics maps for the negative spin {spin} must be the conjugate of the template systematics maps for the positive spin {-spin}'
                )
            if spin in parameters_template_systematics and -spin in parameters_template_systematics:
                assert np.max(np.abs(
                    parameters_template_systematics[spin].imag +
                    parameters_template_systematics[-spin].imag
                )) < precision, f'The parameters of the template systematics maps for the negative spin {spin} must be the conjugate of the parameters of the template systematics maps for the positive spin {-spin}'

    def _get_systematics_maps_chunk(
        self,
        spin_systematics_maps: dict | None,
        dict_utils_template_systematics_maps: dict | None,
        slice_1d_final_map: np.ndarray,
        dtype: np.dtype = np.complex128
    ) -> Spin_maps | None:
        """
        Internal function to compute the systematics maps of a chunk, 
        given the systematics maps and the template systematics maps.
        Systematics maps of the output pixels of a chunk, ``spin -> (n_det, npix_chunk)``:
        the systematics maps ``spin_systematics_maps`` plus the systematics 
        ``sum_t parameters[spin][t, d] * template[spin][t, p]`` built from the 
        templates (one template: parameters ``(n_det,)`` and template ``(npix,)``).
        """

        spins_systematics = set() | (
            set(spin_systematics_maps.spins) if spin_systematics_maps is not None else set()
        ) | (
            set(dict_utils_template_systematics_maps['template_spin_systematics_maps'].spins) if dict_utils_template_systematics_maps is not None else set()
        )

        _check_spin_or_conj = lambda spin, spin_maps_to_check: (
            spin in spin_maps_to_check or -spin in spin_maps_to_check
        )

        systematics_maps_chunk = Spin_maps()
        for spin in spins_systematics:
            if spin < 0:
                continue
            
            # Systematics map of the spin on the pixels of the chunk: the map given, plus the
            # systematics built from the templates; not stored when there is neither
            systematics_map = None
            if spin_systematics_maps is not None and _check_spin_or_conj(spin, spin_systematics_maps):
                systematics_map = spin_systematics_maps[spin][..., slice_1d_final_map]

            if dict_utils_template_systematics_maps is not None and _check_spin_or_conj(spin, dict_utils_template_systematics_maps['template_spin_systematics_maps']):
                template = dict_utils_template_systematics_maps['template_spin_systematics_maps'][spin][..., slice_1d_final_map]
                parameters = dict_utils_template_systematics_maps['parameters_template_systematics'][spin] if spin in dict_utils_template_systematics_maps['parameters_template_systematics'] else np.conj(dict_utils_template_systematics_maps['parameters_template_systematics'][-spin])
                # contribution_from_template_systematics = systematics_maps_chunk[spin] + contract(
                systematics_from_templates = contract(
                    '...d,...p->dp',
                    parameters,
                    np.asarray(template),
                ) #+ contribution_from_spin_systematics
            
            # if contribution_from_template_systematics.size > 0:
            #     systematics_maps_chunk[spin] = contribution_from_template_systematics
            systematics_map = systematics_from_templates if systematics_map is None else systematics_map + systematics_from_templates

            if systematics_map is not None:
                systematics_maps_chunk[spin] = systematics_map
        return systematics_maps_chunk

    def get_interpolated_h_maps_chunk(
        self,
        h_n_spin_dict: Spin_maps,
        slice_1d_hmap: np.ndarray,
        dict_utils_interpolation_det: dict,
        dtype: np.dtype = np.complex128,
    ) -> Spin_maps:
        r"""
        $h$-maps of all the detectors on some pixels, the $h$-maps of the detectors without
        provided $h$-maps being interpolated over the focal plane.

        The hits and the averages $h_n / h_0$ ($n \ge 1$) of the known detectors are
        interpolated, wafer by wafer, with
        :func:`~smarties.utils.interpolation.perform_interpolation_scipy` (linear
        interpolation, nearest known detector outside their convex hull) and stacked after
        the known ones. The $h$-maps of all the detectors are then reweighted by the hits
        with :func:`~smarties.utils.tools.reweight_hmaps_by_hits`: $h^d_0 = $ hits$_d /
        \sum_{d'}$ hits$_{d'}$ and $h^d_n = h^d_0 \times$ average$^d_n$.

        Parameters
        ----------
        h_n_spin_dict: dict or Spin_maps
            $h$-maps of the known detectors, ``n -> (n_known, n_h_map_pixels)``.
        slice_1d_hmap: np.ndarray
            Columns of the $h$-maps (compressed $h$-map pixels) to interpolate.
        dict_utils_interpolation_det: dict
            Dictionary containing the information for the interpolation:
            * ``'dict_positions_detector_known'``: positions of the known detectors, 
            assumed to be in the same order as the $h$-maps in ``h_n_spin_dict``;
            * ``'dict_positions_detector_unknown'``:
              positions of the detectors to interpolate;
            * ``'dict_key_positions'``: ``{'position_x_key': ..., 'position_y_key': ...}``,
              keys of the positions in the two dictionaries above;
            * ``'reweight_by_hits'``: boolean, whether the $h$-maps given are
              reweighted by the hits (normalised, $\sum_d h^d_0 = 1$, as read with
              ``reweight_by_hits=True``). If True, the hits $h^d_0 H$ ( rebuilt from 
              ``H = h_n_spin_dict.total_hits_map``) and the averages $h^d_n / h^d_0$ 
              of the known detectors are interpolated. If False, the $h$-maps given 
              must be the hits and the averages themselves (as read with ``reweight_by_hits=False``), 
              and are interpolated as they are. In both cases, the $h$-maps of 
              all the detectors are then reweighted by the hits with :func:`~smarties.utils.tools.reweight_hmaps_by_hits`.
            * ``'wafer_list_detector_known'`` (optional): list of the wafers of the known detectors, shape ``(n_known,)``. The ordering must be the same as in ``dict_positions_detector_known``. If provided, the interpolation is performed separately for each wafer.
            * ``'wafer_list_detector_unknown'`` (optional): list of the wafers of the detectors to interpolate, shape ``(n_unknown,)``. The ordering must be the same as in ``dict_positions_detector_unknown``. If provided, the interpolation is performed separately for each wafer.

            The interpolation is linear, from at least three known detectors (on the wafer),
            the values outside the convex hull of the known detectors being those of the
            nearest one. The interpolation is performed separately for each wafer, if the
            wafer lists are provided; otherwise all the detectors are interpolated together.
        dtype: np.dtype
            Complex data type of the returned $h$-maps.

        Returns
        -------
        Spin_maps
            $h$-maps ``n -> (n_known + n_interpolated, slice_1d_hmap.size)`` for $n \ge 0$,
            known detectors first.
        """
        dict_key_positions = dict_utils_interpolation_det['dict_key_positions']
        
        position_keys = (dict_key_positions['position_x_key'], dict_key_positions['position_y_key'])

        number_known = np.size(dict_utils_interpolation_det['dict_positions_detector_known'][position_keys[0]])
        number_detectors = number_known + np.size(dict_utils_interpolation_det['dict_positions_detector_unknown'][position_keys[0]])

        list_spin_positive = sorted({abs(int(spin)) for spin in h_n_spin_dict.keys()})

        # Values of the known detectors on the pixels: with reweight_by_hits, their hits (h_0
        # times the total hits) and their averages h_n / h_0; otherwise the h-maps given
        h_maps_spin_0_sliced = h_n_spin_dict[0][:, slice_1d_hmap].real
        if dict_utils_interpolation_det['reweight_by_hits']:
            assert h_n_spin_dict.total_hits_map is not None, 'reweight_by_hits needs the total hits of the h-maps, h_n_spin_dict.total_hits_map'
            inverse_weights = np.where(
                h_maps_spin_0_sliced != 0, 
                1 / np.where(h_maps_spin_0_sliced != 0, h_maps_spin_0_sliced, 1), 
                0
            )
            inverse_weight_spin_0 = h_n_spin_dict.total_hits_map[slice_1d_hmap]
            
            inverse_weight_func = lambda spin: inverse_weight_spin_0 if spin == 0 else inverse_weights
        else:
            inverse_weight_func = lambda spin: 1.

        # h-maps of all the detectors on the pixels, the known detectors first, then the
        # interpolated ones (the order of the positions, of polar_angle and of the efficiencies)
        real_dtype = np.finfo(dtype).dtype
        h_map_interpolated_output = Spin_maps()
        for spin in list_spin_positive:
            if spin == 0:
                sliced_h_map = h_maps_spin_0_sliced
                dtype_to_use = real_dtype
            else:
                sliced_h_map = h_n_spin_dict[spin][:, slice_1d_hmap]
                dtype_to_use = dtype

            h_map_interpolated_output[spin] = np.zeros((number_detectors, slice_1d_hmap.size), dtype=dtype_to_use)
            h_map_interpolated_output[spin][:number_known] = sliced_h_map * inverse_weight_func(spin)

        # Interpolation over each wafer separately, from the known detectors of the same wafer
        # (all the detectors together without wafer lists)
        wafers_known = np.asarray(dict_utils_interpolation_det.get(
            'wafer_list_detector_known', 
            np.zeros(number_known)
        ))
        wafers_unknown = np.asarray(dict_utils_interpolation_det.get(
            'wafer_list_detector_unknown', 
            np.zeros(number_detectors - number_known)
        ))

        for wafer in np.unique(wafers_unknown):

            indices_known = np.flatnonzero(wafers_known == wafer)
            indices_unknown = np.flatnonzero(wafers_unknown == wafer)
    
            assert indices_known.size >= 3, f'The number of known detector on the wafer {wafer} to interpolate from must be at least 3, but it is {indices_known.size}'

            interpolated = perform_interpolation_scipy(
                dictionary_detector_to_interpolate={key: np.asarray(dict_utils_interpolation_det['dict_positions_detector_unknown'][key])[indices_unknown] for key in position_keys},
                dictionary_known_values={
                    **{key: np.asarray(dict_utils_interpolation_det['dict_positions_detector_known'][key])[indices_known] for key in position_keys},
                    **{spin: h_map[indices_known] for spin, h_map in h_map_interpolated_output.items()},
                },
                method_interpolation='linear',
                return_as_spin_maps=False,
                **dict_key_positions,
            )
            for spin, h_map_result in interpolated.items():
                h_map_interpolated_output[spin][number_known + indices_unknown] = h_map_result

        # Reweighting by the hits of all the detectors: h_0 = hits / sum of the hits, h_n = averages * h_0
        reweight_hmaps_by_hits(h_map_interpolated_output, in_place=True)
        return h_map_interpolated_output

    def compute_total_maps_chunks(
            self,
            geometry_dictionary: dict,
            h_n_spin_dict: Spin_maps,
            spin_sky_maps: dict | Spin_maps,
            spin_systematics_maps: dict | Spin_maps = None,
            mask: np.ndarray = None,
            inverse_mapmaking_matrix: np.ndarray = None,
            perform_det_interpolation: bool = False,
            dict_utils_interpolation_det: dict = None,
            return_Q_U: bool = False,
            return_inverse_mapmaking_matrix: bool = False,
            polar_angle: np.ndarray = None,
            number_chunks: int | tuple = 10,
            build_systematics_maps_from_templates: bool = False,
            dict_utils_template_systematics_maps: dict = None,
            single_precision: bool = False,
            polar_efficiency_coeff: np.ndarray = None,
        ):
        r"""
        Compute the total maps from the $h$-maps, the spin sky maps and the spin systematics
        maps, chunk by chunk, for CAR or HEALPix maps, with $h$-maps possibly at a coarser
        resolution than the output maps and possibly interpolated to more detectors.

        The result is the one of
        :meth:`~smarties.mapmaking.FrameworkSystematics.compute_total_maps` with the $h$-maps
        of all the detectors (known and interpolated) projected to the output pixels.

        Parameters
        ----------
        geometry_dictionary: dict
            Geometry of the mapmaking, with ``'projection_pixel'``: ``'car'`` or ``'healpix'``.

            * CAR: ``'shape_car'``, shape of the full output map (default: ``map_shape``);
              ``'wcs_car'``, WCS of the output map, the output maps being ``enmap.ndmap``.
            * HEALPix: ``'nside'`` (default from the mask or ``map_shape``);
              ``'nest'`` (optional), ordering of both maps, NESTED if True,
              RING otherwise (default).

            The output resolution must be an integer multiple of the $h$-map resolution: CAR
            shapes multiple along each axis, nside multiple by a power of 2.
        h_n_spin_dict: dict or Spin_maps
            Dictionary of the $h$-maps, ``n -> (n_det, n_h_map_pixels)``, on the pixels of the
            $h$-map grid containing at least one observed output pixel, in increasing order.
            Normalised, with $\sum_d h_0 = 1$ (over the known detectors with
            ``perform_det_interpolation``). Only $n \ge 0$ is needed. Never copied.
        spin_sky_maps: dict or Spin_maps
            Dictionary of the spin sky maps on the observed output pixels, ``spin -> (npix,)``
            (e.g. if nstokes=3, the keys are 0, -2, 2 and the fields (I, (Q-iU)/2, (Q+iU)/2)).
        spin_systematics_maps: dict or Spin_maps (optional)
            Dictionary of the spin systematics maps on the observed output pixels,
            ``spin -> (n_det, npix)``, for all the detectors (known first). If None, no
            systematics maps are considered.
        mask: np.ndarray (optional)
            Mask of the full output map; only the pixels where it is non-zero are computed.
            Default: all the pixels.
        inverse_mapmaking_matrix: np.ndarray (optional)
            Inverse of the mapmaking matrix on the observed output pixels, shape
            ``(npix, nstokes, nstokes)``. If None, it is computed from the $h$-maps.
        perform_det_interpolation: bool (optional)
            If True, interpolate the $h$-maps of the detectors without provided $h$-maps
            from those of the known detectors (see :meth:`get_interpolated_h_maps_chunk`).
        dict_utils_interpolation_det: dict (optional)
            Information for the interpolation, see :meth:`get_interpolated_h_maps_chunk`.
        return_Q_U: bool (optional)
            If True, return I, Q, U maps instead of spin maps. Default is False.
        return_inverse_mapmaking_matrix: bool (optional)
            If True, also return the inverse of the mapmaking matrix. Default is False.
        polar_angle: np.ndarray (optional)
            Polarisation angles of all the detectors in radians (known detectors first, then
            interpolated ones). Default: 0.
        number_chunks: int or tuple (optional)
            CAR: number of chunks along each axis of the $h$-map grid, a pair ``(n_y, n_x)``,
            or a scalar ``n`` giving ``int(sqrt(n)) + 1`` chunks along each axis. HEALPix:
            number of chunks of observed $h$-map pixels. Default is 10.
        build_systematics_maps_from_templates: bool (optional)
            If True, add the systematics ``sum_t parameters[spin][t, d] * template[spin][t, p]``
            given in ``dict_utils_template_systematics_maps``. Default is False.
        dict_utils_template_systematics_maps: dict (optional)
            ``'template_spin_systematics_maps'``: Spin_maps of the templates on the observed
            output pixels, ``spin -> (npix,)`` or ``(n_templates, npix)``;
            ``'parameters_template_systematics'``: dictionary of the detector parameters,
            ``spin -> (n_det,)`` or ``(n_templates, n_det)``. See
            :class:`smarties.systematics.templates.TemplateSystematics`, which combines several
            systematics in this format.
        single_precision: bool (optional)
            If True, compute in single precision. Default is False.
        polar_efficiency_coeff: np.ndarray (optional)
            Polarisation efficiencies of all the detectors. Default: 1.

        Returns
        -------
        Spin_maps or np.ndarray
            Output spin maps on the observed pixels (positive spins, the negative ones being
            their conjugates), or the ``(nstokes, npix)`` I, Q, U maps if ``return_Q_U``;
            with ``return_inverse_mapmaking_matrix``, a tuple
            ``(maps, inverse_mapmaking_matrix)``.
        """

        # Few assert
        assert 'projection_pixel' in geometry_dictionary.keys(), 'The projection pixelization should be specified in the geometry dictionary, with key "projection_pixel" and element "healpix" or "car"'
        projection_pixel = geometry_dictionary['projection_pixel']
        assert projection_pixel in ['healpix', 'car'], f'Unknown projection pixel: {projection_pixel}'

        # Geometry of the output map and of the h-maps
        if projection_pixel == 'car':
            shape_final_map = geometry_dictionary.get('shape_car', self.map_shape)
            assert len(shape_final_map) == 2, 'For CAR pixelization, the shape of the final map must be 2D'
            wcs_car_final_map = geometry_dictionary['wcs_car']
            shape_hmap = h_n_spin_dict.shape_fullsky
            assert len(shape_hmap) == 2, 'The shape of the downgraded map must be 2D for CAR pixelization'
            nest = False
        else:
            shape_final_map = (hp.nside2npix(int(geometry_dictionary['nside'])),)
            shape_hmap = h_n_spin_dict.shape_fullsky
            nside_h_maps = hp.npix2nside(int(np.prod(shape_hmap)))
            assert nside_h_maps & (nside_h_maps - 1) == 0, 'The nside of the h-maps must be a power of 2'

            wcs_car_final_map = None
            nest = bool(geometry_dictionary.get('nest', False))

        if mask is None:
            mask = np.ones(shape_final_map, dtype=np.int8)
        else:
            assert np.size(mask) == np.prod(shape_final_map), 'The mask must cover the full output map'
            
        npix = np.count_nonzero(mask)
        assert npix > 0, 'The mask must contain at least one observed pixel'

        dtype = np.complex64 if single_precision else np.complex128
        precision = 1e-6 if single_precision else 1e-14

        self._check_h_maps_consistency(
            h_n_spin_dict, 
            perform_det_interpolation=perform_det_interpolation,
            precision=precision,
        )
        
        time_start = time.time()

        list_spin_h_maps = sorted({abs(int(spin)) for spin in h_n_spin_dict.keys()})
        assert 0 in list_spin_h_maps, 'The h_0 maps must be provided'

        if spin_systematics_maps is None and dict_utils_template_systematics_maps is None:
            logger.warning("No systematics maps provided, assuming no systematics")

        if spin_systematics_maps is None:
            logger.warning("No systematics maps provided, assuming no systematics")
            spin_systematics_maps = Spin_maps.from_dictionary({})

        self._check_spin_maps_sky_systematics(
            spin_sky_maps = spin_sky_maps, 
            spin_systematics_maps = spin_systematics_maps, 
            dtype = dtype,
            npix = npix
        )

        set_spins_maps = set(np.unique(spin_sky_maps.spins + spin_systematics_maps.spins))
        if dict_utils_template_systematics_maps is not None:
            set_spins_maps |= set(np.unique(dict_utils_template_systematics_maps['template_spin_systematics_maps'].spins))

        # Basic checkson the interpolation of the h_n maps to more detectors
        total_number_detectors = np.shape(h_n_spin_dict[0])[0]
        if perform_det_interpolation:
            total_number_detectors = self._check_interpolation(dict_utils_interpolation_det)
            logger.info(f'Interpolating the h_n maps from {np.shape(h_n_spin_dict[0])[0]} to {total_number_detectors} detectors')
            

        if build_systematics_maps_from_templates:
            logger.info("Template systematics maps provided, checking their consistency...")
            self._check_template_systematics_maps(
                dict_utils_template_systematics_maps,
                dtype = dtype
            )
            set_spins_maps = set(np.unique(spin_sky_maps.spins + spin_systematics_maps.spins))

        else:
            dict_utils_template_systematics_maps = None

        spin_h_maps = sorted(set(h_n_spin_dict.spins) | {-s for s in h_n_spin_dict.spins if s > 0})

        self._check_consistency_spin_maps(
            set_spin_maps = set_spins_maps,
            h_n_spin_dict = h_n_spin_dict,
        )

        all_spins = sorted(set_spins_maps | set(spin_h_maps) | set(self.list_spin_output))

        # Polarisation angles and efficiencies
        if polar_angle is not None:
            polar_angle = np.asarray(polar_angle)
            assert polar_angle.size == total_number_detectors, 'The polar angle map must have shape (n_det,) accounting first for detectors for which h-maps are known, and then the h-maps to be interpolated, with the same ordering as the detector locations'
        
        polar_angle_coeff = {
            spin: np.ones(total_number_detectors, dtype=dtype) if polar_angle is None else np.exp(spin * 1j * polar_angle).astype(dtype)
            for spin in all_spins
        }

        if polar_efficiency_coeff is None:
            polar_efficiency_coeff = np.ones(total_number_detectors)
        else:
            polar_efficiency_coeff = np.asarray(polar_efficiency_coeff)
            assert polar_efficiency_coeff.size == total_number_detectors, (
                'The polar efficiency coefficients must have shape (n_det,), including the detectors to be interpolated'
            )

        
        ratio_resolution_change = get_ratio_resolution(projection_pixel, shape_final_map, shape_hmap)
        perform_resolution_change = any(ratio != 1 for ratio in ratio_resolution_change)
        if perform_resolution_change:
            logger.info(f"Performing resolution change from {shape_hmap} to {shape_final_map} with ratio {ratio_resolution_change}...")

        
        
        observed_pixels_final_map = np.flatnonzero(np.asarray(mask))
        npix = observed_pixels_final_map.size

        mask_h_maps = downgrade_mask(mask, projection_pixel, shape_final_map, shape_hmap, nest=nest)

        observed_pixels_h_maps = np.flatnonzero(mask_h_maps)
        number_pixels_h_maps = np.shape(h_n_spin_dict[0])[-1]
        assert number_pixels_h_maps == observed_pixels_h_maps.size, f'The h-maps have {number_pixels_h_maps} pixels, but the h-map grid has {observed_pixels_h_maps.size} pixels containing observed output pixels: the h-maps must be given on these pixels'

        number_spins = len(self.list_spin_output)
        if inverse_mapmaking_matrix is not None:
            assert np.ndim(inverse_mapmaking_matrix) == 3 and np.shape(inverse_mapmaking_matrix)[1:] == (number_spins, number_spins), 'The inverse mapmaking matrix must be a 3D array of shape (n_pix, n_stokes, n_stokes)'
            assert np.shape(inverse_mapmaking_matrix)[0] == npix, 'The inverse mapmaking matrix must be given for the observed output pixels'
            # Without resolution change, the h-map and output pixels are the same when their numbers are equal

        # Retrieving chunks of the h-map grid
        if projection_pixel == 'car':
            list_chunks_downgraded_map = get_chunks_from_shape(shape_hmap, number_chunks=number_chunks)
        else:
            list_chunks_downgraded_map = get_chunks_healpix(mask_h_maps, number_chunks=int(np.prod(number_chunks)), nest=nest)

        logger.info(
            f"Mapmaking per chunk: {npix} pixels, {number_pixels_h_maps} h-map pixels, {total_number_detectors} detectors, {len(list_chunks_downgraded_map)} chunks"
        )

        # final_sky_fields = np.zeros((number_spins, npix), dtype=dtype)
        get_final_map = lambda x: x if wcs_car_final_map is None else enmap.ndmap(x, wcs=wcs_car_final_map)
        dict_final_sky_fields = Spin_maps.from_dictionary(
            {
                spin: 
                get_final_map(np.zeros((npix,), dtype=dtype)) 
                for spin in self.list_spin_output if spin >= 0
            }
        )
        dict_final_sky_fields.set_projection_pixel(projection_pixel)
        dict_final_sky_fields.set_shape_fullsky(shape_final_map)
        if projection_pixel == 'car':
            dict_final_sky_fields.set_wcs_car(wcs_car_final_map)
        
        if return_inverse_mapmaking_matrix and inverse_mapmaking_matrix is None:
            final_inverse_mapmaking_matrix = np.zeros(
                (npix, number_spins, number_spins), dtype=dtype
            )
        else:
            final_inverse_mapmaking_matrix = inverse_mapmaking_matrix

        for chunk_elem in list_chunks_downgraded_map:

            # Observed output pixels of the chunk, h-map pixels containing them, and projector
            pixels_chunk = get_pixels_chunk(projection_pixel, chunk_elem, shape_final_map, shape_hmap, nest=nest)
            slice_1d_final_map, slice_1d_hmap, projector_h_n = get_projector_chunk(
                projection_pixel,
                pixels_chunk,
                observed_pixels_final_map,
                observed_pixels_h_maps,
                shape_final_map,
                shape_hmap,
                nest=nest,
            )
            if slice_1d_final_map.size == 0:
                continue
            logger.info(f"Processing chunk with {slice_1d_final_map.size} pixels and {slice_1d_hmap.size} h-map pixels...")

            # h-maps of the chunk, never copied: the h-maps given, read through the columns
            # slice_1d_hmap of the h-map pixels of the chunk, or the h-maps interpolated over the
            # focal plane on these pixels. columns_output: column of the h-maps of each output pixel
            if perform_det_interpolation:
                h_n_spin_chunk = self.get_interpolated_h_maps_chunk(
                    h_n_spin_dict, slice_1d_hmap, dict_utils_interpolation_det, dtype=dtype
                )
                columns_h_maps, columns_output = None, projector_h_n
            else:
                h_n_spin_chunk = h_n_spin_dict
                columns_h_maps, columns_output = slice_1d_hmap, slice_1d_hmap[projector_h_n]
            npix_h_maps_chunk = slice_1d_hmap.size

            # Inverse mapmaking matrix of the chunk, and the row of each output pixel: given at the
            # output resolution, or computed on the pixels of the h-maps of the chunk
            if inverse_mapmaking_matrix is not None:
                inverse_mapmaking_matrix_chunk, rows_inverse = inverse_mapmaking_matrix, slice_1d_final_map
            else:
                inverse_mapmaking_matrix_chunk = self.get_inverse_mapmaking_matrix(
                    h_n_spin_chunk,
                    npix_h_maps_chunk,
                    polar_angle_coeff=polar_angle_coeff,
                    polar_efficiency_coeff=polar_efficiency_coeff,
                    slice_to_apply=columns_h_maps,
                )
                rows_inverse = projector_h_n 

            # compute_spin_convolution (numba) reads the maps of size 1 as zero placeholders: the
            # output pixel of a chunk with a single one is computed twice (only its indices are repeated)
            repeat = 2 if slice_1d_final_map.size == 1 else 1
            pixels_mapmaking = np.repeat(slice_1d_final_map, repeat)

            # Data vector of the chunk, with zero placeholders for the spins missing in the sky or
            # systematics maps (as compute_total_maps)
            spin_sky_maps_chunk = Spin_maps.from_dictionary(
                {spin: value[pixels_mapmaking] for spin, value in spin_sky_maps.items()}
            )
            spin_systematics_maps_chunk = self._get_systematics_maps_chunk(
                spin_systematics_maps = spin_systematics_maps, 
                dict_utils_template_systematics_maps = dict_utils_template_systematics_maps, 
                slice_1d_final_map = pixels_mapmaking,
                dtype = dtype
            )
            
            spin_coupled_maps = self.compute_spin_convolution(
                spin_sky_maps = spin_sky_maps_chunk,
                spin_systematics_maps = spin_systematics_maps_chunk,
                h_n_spin_dict = h_n_spin_chunk,
                npix = pixels_mapmaking.size,
                polar_angle_coeff = polar_angle_coeff,
                polar_efficiency_coeff = polar_efficiency_coeff,
                dtype = dtype,
                projector_h_n = np.repeat(columns_output, repeat),
                return_array = True,
            )[::repeat]

            # Output maps of the chunk, as compute_total_maps
            for i, spin in enumerate(self.list_spin_output):
                if spin < 0:
                    continue

                # dict_final_sky_fields[spin][slice_1d_final_map] = (
                #     (inverse_mapmaking_matrix_chunk[rows_inverse,i,:]*spin_coupled_maps).sum(axis=-1)
                # )
                output_chunk = np.zeros(slice_1d_final_map.size, dtype=dtype)
                _numba_apply_pixel_matrix(
                    matrix_line=inverse_mapmaking_matrix_chunk[rows_inverse,i,:],
                    vector=spin_coupled_maps,
                    output=output_chunk,
                )
                dict_final_sky_fields[spin][slice_1d_final_map] = output_chunk

            if return_inverse_mapmaking_matrix and inverse_mapmaking_matrix is None:
                final_inverse_mapmaking_matrix[slice_1d_final_map] = inverse_mapmaking_matrix_chunk[rows_inverse]

            # Arrays of the chunk released before those of the next chunk are built
            del h_n_spin_chunk, inverse_mapmaking_matrix_chunk, spin_sky_maps_chunk, spin_systematics_maps_chunk, spin_coupled_maps

        logger.info(f"Mapmaking per chunk done in {time.time() - time_start:.2f} seconds")

        if return_Q_U:
            final_Q = (dict_final_sky_fields[-2] + dict_final_sky_fields[2])/2.
            final_U = 1j*(dict_final_sky_fields[-2] - dict_final_sky_fields[2])/2.
            if self.nstokes == 3:
                final_I = dict_final_sky_fields[0]
                output = np.vstack([final_I, final_Q, final_U])
            else:
                output = np.vstack([final_Q, final_U])
        else:
            output = dict_final_sky_fields

        if return_inverse_mapmaking_matrix:
            return output, final_inverse_mapmaking_matrix
        return output
