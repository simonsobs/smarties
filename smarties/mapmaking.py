# This file is part of SMARTIES.
# Copyright (c) 2024-2026 bers of the Simons Simons Observatory Collaboration.
# lease refer to the LICENSE file in the root of this repository.

from copy import deepcopy
import numpy as np
from opt_einsum import contract
from pixell import enmap

from smarties.utils.tools import get_coupled_spin, get_row_mapmaking_matrix
from smarties.sky.cmb import create_CMB_spin_maps
from smarties.hn import Spin_maps, Spin_nm
from smarties.utils.tools import reweight_hmaps_by_hits
from smarties.utils.interpolation import perform_interpolation_scipy
from smarties.utils.resolution import (
    get_slice_1d_from_2d,
    get_chunks_from_shape,
    get_projector_map_resolution,
    get_chunks_final_map_from_downgraded_chunks
)

__all__ = [
    'FrameworkSystematics'
]

class FrameworkSystematics(object):
    """
    Class to simulate systematics maps
    """

    def __init__(self, map_shape, nstokes, lmax, list_spin_output=[-2,2]):
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
        list_spin: list[int]
            List of spins involved in the signal maps only (for CMB only, the spins are -2, 2 with polarization only)
        """
        assert np.unique(list_spin_output).size == np.array(list_spin_output).size, 'The list of spins must be unique'
        if not np.isin(list_spin_output, np.array([0,-2,2])).all():
            print('The output spins maps appeared to contain spin different than 0, -2 or 2, the package has not been tested in this case!', flush=True)
        self.list_spin_output = list_spin_output # list spins involved in the signal maps only (for CMB only, the spins are 0, -2, 2 if intensity is involved)
        self.nstokes = nstokes
        self.lmax = lmax
        self.map_shape = map_shape

    @property
    def list_spin_input(self):
        if self.nstokes == 3:
            return [0, 2, -2]
        elif self.nstokes == 2:
            return [2, -2]
        else:
            raise NotImplemented('The number of Stokes parameters must be 2 (polarization only) or 3 (intensity and polarization), other cases are not implemented yet')

    def get_spin_sky_maps(self, nside, fwhm=0., seed=42):
        """
        Get the spin CMB maps which are the following for intensity and polarization:
            * Spin 0: I
            * Spin 2: (Q + iU)/2.
            * Spin -2: (Q - iU)/2.

        Parameters
        ----------
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
            seed=seed)

    def get_inverse_mapmaking_matrix(
            self,
            h_n_spin_dict: dict | Spin_maps,
            npix: int = None,
            polar_angle_coeff: np.ndarray = None,
            polar_efficiency_coeff: np.ndarray = None,
            slice_to_apply: slice = None
        ):
        """
        Compute the inverse of the mapmaking matrix from the h_n maps

        Parameters
        ----------
        h_n_spin_dict: dict or Spin_maps
            Dictionary of the summed $h_n$ maps, with the keys being the spins and the values the $h_n$ maps
        npix: int
            Number of pixels in the observed area, default is None, then the number of pixels is inferred from the input maps

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
            mapmaking_matrix[:,i,:] = get_row_mapmaking_matrix(
                reference_spin=reference_spin,
                h_n_spin_dict=h_n_spin_dict,
                list_spin_input=self.list_spin_input,
                dtype=dtype,
                polar_angle_coeff=polar_angle_coeff,
                polar_efficiency_coeff=polar_efficiency_coeff,
                slice_to_apply=slice_to_apply
            )
        # Then, compute the inverse of the mapmaking matrix
        return np.linalg.pinv(mapmaking_matrix)

    def compute_total_maps(
            self,
            mask: np.ndarray,
            h_n_spin_dict: dict | Spin_maps,
            spin_sky_maps: dict | Spin_maps,
            spin_systematics_maps: dict | Spin_maps = None,
            inverse_mapmaking_matrix : np.ndarray = None,
            return_Q_U: bool = False,
            return_inverse_mapmaking_matrix: bool = False,
            mask_input: bool = False,
            polar_angle: np.ndarray = None,
            polar_efficiency_coeff: np.ndarray = None,
            projector_h_n: np.ndarray = None,
            wcs_car=None
        ):
        """
        Compute the total maps from the $h_n$ maps, the spin CMB maps and the spin systematics maps

        Parameters
        ----------
        mask: np.ndarray
            Mask of the maps, only the pixels in the observed area will be considered for the inversion
        h_n_spin_dict: dict or Spin_maps
            Dictionary of the summed $h_n$ maps, with the keys being the spins and the values the $h_n$ maps
        spin_sky_maps: dict or Spin_maps
            Dictionary of the spin CMB maps, with the keys being the spins and the values the spin CMB maps
            (e.g. if nstokes=3, the keys are 0, -2, 2 and the fields (I, Q-iU, Q+iU))
        spin_systematics_maps: dict or Spin_maps (optional)
            Dictionary of the spin systematics maps, with the keys being the spins and the values the spin systematics maps,
            of shape (n_det, npix), with n_det being the number of detectors (1 for spin=0) and npix the number of pixels.
            If None, no systematics are considered (default is None).
        inverse_mapmaking_matrix: np.ndarray (optional)
            The inverse of the mapmaking matrix, with the shape (npix, nstokes, nstokes), with npix being the number of pixels
            in the observed area of the provided mask. If None, the inverse mapmaking matrix will be computed from the provided h_n maps.
            Default is None.
        return_Q_U: bool (optional)
            If True, return the Q and U maps instead of the spin -2 and 2 maps, default is False.
        return_inverse_mapmaking_matrix: bool
            If True, return the inverse of the mapmaking matrix, default is False.
        mask_input: bool (optional)
            If True, the input maps will be copied and masked (WARNING: possibly very memory intensive), otherwise the input maps will not be masked,
            and assumed to be provided in the right format. Default is True.
        polar_angle: np.ndarray (optional)
            The polar angle to use for the mapmaking, if None, the polar angle will be computed from the provided h_n maps.
            Must be provided in radians. Default is None.

        Returns
        -------
        final_CMB_fields: np.ndarray
            the final CMB fields, with the shape (npix, nstokes) if return_Q_U is False, (npix, 3) otherwise
        """

        # Few assert
        print([spin_sky_maps[spin].ndim for spin in spin_sky_maps.keys()])
        assert np.allclose([spin_sky_maps[spin].ndim for spin in spin_sky_maps.keys()], 1), 'The CMB maps must be 1D arrays of shape (n_pix)'
        assert np.allclose([h_n_spin_dict[spin].ndim for spin in h_n_spin_dict.keys() if spin != 0 ], 2), 'The h_n maps must be 2D arrays of shape (n_det, n_pix)'
        if spin_systematics_maps is not None:
            assert np.allclose([spin_systematics_maps[spin].ndim for spin in spin_systematics_maps.keys() if spin != 0 ], 2), 'The systematics maps must be 2D arrays of shape (n_det, n_pix)'

        if projector_h_n is not None:
            assert np.all(
                np.isin(np.unique(projector_h_n), np.arange(h_n_spin_dict[0].shape[-1]))
            ), 'The projector_h_n must contain valid pixel indices'
            assert projector_h_n.size == spin_systematics_maps[list(spin_systematics_maps.keys())[0]].shape[-1], 'The projector_h_n must have the same size as the number of pixels in the systematics maps'
            assert issubclass(
                projector_h_n.dtype.type, np.integer
            ), 'The projector_h_n must be an array of integers'
        else:
            projector_h_n = ... # Use all pixels

        # Check that the h_n maps are normalized
        spin_0 = Spin_nm((0,0)) if np.array(h_n_spin_dict.spins)[0].size == 2 else 0
        assert np.all(np.abs(h_n_spin_dict[spin_0].sum(axis=0) - 1) < 1e-14), 'The h_n maps must be normalized'

        if mask is None:
            mask = np.ones(self.map_shape, dtype=np.int8)

        # Masking the h_n maps, CMB maps and systematics maps
        observed_pixels_array = mask != 0
        if mask_input:
            h_n_spin_dict = Spin_maps.from_dictionary(
                {spin: h_n_spin_dict[spin][...,observed_pixels_array]
                 if np.size(h_n_spin_dict[spin][0,...]) == mask.size
                 else h_n_spin_dict[spin] for spin in h_n_spin_dict.keys()}
            )
            spin_sky_maps = Spin_maps.from_dictionary(
                {spin: spin_sky_maps[spin][...,observed_pixels_array]
                 if np.size(spin_sky_maps[spin]) == mask.size
                 else spin_sky_maps[spin] for spin in spin_sky_maps.keys()}
            )
            if spin_systematics_maps is not None:
                spin_systematics_maps = Spin_maps.from_dictionary(
                    {spin: spin_systematics_maps[spin][...,observed_pixels_array]
                     if np.size(spin_systematics_maps[spin][0,...]) == mask.size
                     else spin_systematics_maps[spin] for spin in spin_systematics_maps.keys()}
                )
        # else:
        #     spin_sky_maps = Spin_maps.from_dictionary(spin_sky_maps)

        if polar_angle is None:
            polar_angle_coeff = {spin: np.ones(h_n_spin_dict[spin].shape[0], dtype=complex) for spin in h_n_spin_dict.spins} # Default is to not apply any polar angle, i.e. the detectors are all aligned in the same direction
        else:
            assert polar_angle.size == h_n_spin_dict[spin_0].shape[0], 'The polar angle map must have the same shape as the h_n maps'

            polar_angle_coeff = {spin: np.exp(spin * 1j * polar_angle) for spin in h_n_spin_dict.spins}
            #TODO: Generalize to m != 0 for HWP angles
        if polar_efficiency_coeff is not None:
            assert polar_efficiency_coeff.size == h_n_spin_dict[spin_0].shape[0], 'The polar efficiency map must have the same shape as the h_n maps'
        else:
            polar_efficiency_coeff = np.ones(h_n_spin_dict[spin_0].shape[0])
        npix = mask[observed_pixels_array].size

        if spin_systematics_maps is None:
            print("No systematics maps provided, assuming no systematics", flush=True)
            spin_systematics_maps = Spin_maps.from_dictionary({spin: np.zeros(1) for spin in spin_sky_maps.spins})

        assert np.all(sum(spin_sky_maps.values()).imag < 1e-14), 'The sum of the input sky maps must be real, the imaginary part is not expected to be non-zero'
        assert np.all(sum(spin_systematics_maps.values()).imag < 1e-14), 'The sum of the input systematics maps must be real, the imaginary part is not expected to be non-zero'


        for spin in set(spin_sky_maps.spins):
            if spin not in spin_systematics_maps:
                spin_systematics_maps[spin] = np.zeros(1) # If a spin is not provided in the systematics maps, we assume that the systematics maps for this spin are zero

        for spin in set(spin_systematics_maps.spins):
            if spin not in spin_sky_maps:
                spin_sky_maps[spin] = np.zeros(1) # If a spin is not provided in the sky maps, we assume that the sky maps for this spin are zero

        if inverse_mapmaking_matrix is None:
            inverse_mapmaking_matrix = self.get_inverse_mapmaking_matrix(
                h_n_spin_dict,
                polar_angle_coeff=polar_angle_coeff,
                polar_efficiency_coeff=polar_efficiency_coeff,
                npix=npix,
            )
        else:
            assert inverse_mapmaking_matrix.shape == (npix, self.nstokes, self.nstokes), 'The inverse mapmaking matrix must be of shape (npix, nstokes, nstokes), with npix being the number of pixels in the observed area of the provided mask'

        print("Finishing the mapmaking process, computing the total maps...", flush=True)
        # Second, form the data vector composed of (<d_j>, <d_j cos 2\phi_j>, <d_j sin 2\phi_j>)

        print("Computing the spin coupled maps...", flush=True)
        spin_coupled_maps = np.zeros((npix, len(self.list_spin_output),), dtype=complex)
        list_spin_maps = list(np.unique(spin_sky_maps.spins + spin_systematics_maps.spins))

        factor_func = lambda spin: 1 if np.sum(spin) == 0 else .5
        # Depends on the definition of the pointing matrix
        def polar_efficiency_func(spin):
            if spin in [-2, 2]:
                return polar_efficiency_coeff
            else:
                return np.ones_like(polar_efficiency_coeff, dtype=int)

        for i, spin in enumerate(self.list_spin_input):
            # Get all combinations of spins (k-k', k') such that k-k' = spin
            coupled_spins = get_coupled_spin(
                reference_spin=spin,
                available_h_n_spin=h_n_spin_dict.spins,
                available_signal_spins=list_spin_maps
            )

            print(f'Coupled spins for spin {spin}: {coupled_spins}', flush=True)

            # \sum_{k' = -\infty}^{\infty} h_{k-k'} S_{k'} on all (k-k', k') pairs
            for tuple_spins in coupled_spins:
                spin_coupled_maps[...,i] += factor_func(spin) * contract(
                    'd,d,d...,d...->...',
                    polar_efficiency_func(spin),
                    polar_angle_coeff[spin],
                    h_n_spin_dict[tuple_spins[0]][projector_h_n],
                    spin_systematics_maps[tuple_spins[1]]
                    + contract(
                        'd,...->d...',
                        polar_angle_coeff[-tuple_spins[1]],
                        spin_sky_maps[tuple_spins[1]]
                    ) # Polarization angle to remove for consistent modeling of the data
                )

        get_final_map = lambda x: x if wcs_car is None else enmap.ndmap(
            x,
            wcs=wcs_car
        )

        print("Computing the final CMB fields...", flush=True)
        # Finally, compute the final CMB fields
        final_CMB_fields = get_final_map(
            contract(
                '...ij,...j->i...',
                inverse_mapmaking_matrix[projector_h_n],
                spin_coupled_maps
            )
        )

        print("Final CMB fields computed, transforming them into Spin_maps...", flush=True)
        dict_final_CMB_fields = Spin_maps.from_list_maps(final_CMB_fields, self.list_spin_output)

        if return_Q_U:
            final_Q = (dict_final_CMB_fields[-2] + dict_final_CMB_fields[2])/2.
            final_U = 1j*(dict_final_CMB_fields[-2] - dict_final_CMB_fields[2])/2.
            if self.nstokes == 3:
                final_I = dict_final_CMB_fields[0]
                output = np.vstack([final_I, final_Q, final_U])
            else:
                output = np.vstack([final_Q, final_U])
        else:
            output = dict_final_CMB_fields

        if return_inverse_mapmaking_matrix:
            return output, inverse_mapmaking_matrix
        return output


    def compute_total_maps_chunks(
            self,
            mask: np.ndarray,
            geometry_dictionary: dict,
            h_n_spin_dict: dict | Spin_maps,
            spin_sky_maps: dict | Spin_maps,
            spin_systematics_maps: dict | Spin_maps = None,
            inverse_mapmaking_matrix : np.ndarray = None,
            perform_det_interpolation: bool = False,
            dict_utils_interpolation_det: dict = None,
            return_Q_U: bool = False,
            return_inverse_mapmaking_matrix: bool = False,
            polar_angle: np.ndarray = None,
            number_chunks: int = 10,
            build_systematics_maps_from_templates: bool = False,
            dict_utils_template_systematics_maps: dict = None,
        ):
        """
        Compute the total maps from the $h_n$ maps, the spin CMB maps and the spin systematics maps

        Parameters
        ----------
        mask: np.ndarray
            Mask of the maps, only the pixels in the observed area will be considered for the inversion
        h_n_spin_dict: dict or Spin_maps
            Dictionary of the summed $h_n$ maps, with the keys being the spins and the values the $h_n$ maps
        spin_sky_maps: dict or Spin_maps
            Dictionary of the spin CMB maps, with the keys being the spins and the values the spin CMB maps
            (e.g. if nstokes=3, the keys are 0, -2, 2 and the fields (I, Q-iU, Q+iU))
        spin_systematics_maps: dict or Spin_maps (optional)
            Dictionary of the spin systematics maps, with the keys being the spins and the values the spin systematics maps,
            of shape (n_det, npix), with n_det being the number of detectors (1 for spin=0) and npix the number of pixels.
            If None, no systematics are considered (default is None).
        inverse_mapmaking_matrix: np.ndarray (optional)
            The inverse of the mapmaking matrix, with the shape (npix, nstokes, nstokes), with npix being the number of pixels
            in the observed area of the provided mask. If None, the inverse mapmaking matrix will be computed from the provided h_n maps.
            Default is None.
        return_Q_U: bool (optional)
            If True, return the Q and U maps instead of the spin -2 and 2 maps, default is False.
        return_inverse_mapmaking_matrix: bool
            If True, return the inverse of the mapmaking matrix, default is False.
        mask_input: bool (optional)
            If True, the input maps will be copied and masked (WARNING: possibly very memory intensive), otherwise the input maps will not be masked,
            and assumed to be provided in the right format. Default is True.
        polar_angle: np.ndarray (optional)
            The polar angle to use for the mapmaking, if None, the polar angle will be computed from the provided h_n maps.
            Must be provided in radians. Default is None.

        Returns
        -------
        final_CMB_fields: np.ndarray
            the final CMB fields, with the shape (npix, nstokes) if return_Q_U is False, (npix, 3) otherwise

        Notes
        -----
        When using interpolation of the h_n maps, the interpolation is performed for the positive spins and the negative spin maps are obtained by conjugation, which is expected to be the case for the h_n maps. The interpolation is performed with the `perform_interpolation_scipy` function, which uses the `griddata` function from scipy for the interpolation. The weights for the interpolation are currently set to 1 for all detectors, but they can be changed by providing a list of weights in the parameters of the function. The error precision for the interpolation can also be set in the parameters of the function.
        """

        # Few assert
        assert 'projection_pixel' in geometry_dictionary.keys(), 'The projection pixelization should be specified in the geometry dictionary, with key "projection_pixel" and element "healpix" or "car"'
        projection_pixel = geometry_dictionary['projection_pixel']
        assert projection_pixel in ['healpix', 'car'], f'Unknown projection pixel: {projection_pixel}'

        assert np.allclose([spin_sky_maps[spin].ndim for spin in spin_sky_maps.keys()], 1), 'The CMB maps must be 1D arrays of shape (n_pix)'
        assert np.allclose([h_n_spin_dict[spin].ndim for spin in h_n_spin_dict.keys() if spin != 0 ], 2), 'The h_n maps must be 2D arrays of shape (n_det, n_pix)'
        if spin_systematics_maps is not None:
            assert np.allclose([spin_systematics_maps[spin].ndim for spin in spin_systematics_maps.keys() if spin != 0 ], 2), 'The systematics maps must be 2D arrays of shape (n_det, n_pix)'



        # Check that the h_n maps are normalized
        spin_0 = Spin_nm((0,0)) if np.array(h_n_spin_dict.spins)[0].size == 2 else 0
        if not perform_det_interpolation:
            assert np.all(np.abs(h_n_spin_dict[spin_0].sum(axis=0) - 1) < 1e-14), 'The h_n maps must be normalized'

        if mask is None:
            if projection_pixel == 'car':
                assert np.isin(
                    ['wcs_car_final_map'],
                    list(geometry_dictionary.keys())
                ).all(), 'For CAR pixelization, the wcs of the final map must be provided in the geometry dictionary'
                mask = enmap.ones(self.map_shape, wcs=geometry_dictionary['wcs_car_final_map'])
            elif projection_pixel == 'healpix':
                mask = np.ones(self.map_shape, dtype=np.int8)

        npix = mask[mask != 0].size

        if inverse_mapmaking_matrix is not None:
            assert inverse_mapmaking_matrix.ndim == 3, 'The inverse mapmaking matrix must be a 3D array of shape (n_pix, n_stokes, n_stokes)'
            assert (perform_det_interpolation and perform_resolution_change == False) and inverse_mapmaking_matrix.shape == (npix, self.nstokes, self.nstokes), 'Interpolation and resolution change of the mapmaking matrix are implemetend and the inverse mapmaking matrix will be recomputed, so it should not be provided as an input'
            #TODO: Change as a warning

        if polar_angle is None:
            polar_angle_coeff = {spin: np.ones(h_n_spin_dict[spin].shape[0], dtype=complex) for spin in h_n_spin_dict.spins} # Default is to not apply any polar angle, i.e. the detectors are all aligned in the same direction
        else:
            if not perform_det_interpolation:
                assert polar_angle.size == h_n_spin_dict[spin_0].shape[0], 'The polar angle map must have the same shape as the h_n maps'
            else:
                assert polar_angle.ndim == 1, 'The polar angle map must have shape (n_det,) accounting first for detectors for which h-maps are known, and then the h-maps to be interpolated, with the same ordering as the detector locations'

            polar_angle_coeff = {spin: np.exp(spin * 1j * polar_angle) for spin in h_n_spin_dict.spins}
            #TODO: Generalize to m != 0 for HWP angles

        if spin_systematics_maps is None:
            print("No systematics maps provided, assuming no systematics", flush=True)
            spin_systematics_maps = Spin_maps.from_dictionary({spin: np.zeros(1) for spin in spin_sky_maps.spins})

        assert np.all(sum(spin_sky_maps.values()).imag < 1e-14), 'The sum of the input sky maps must be real, the imaginary part is not expected to be non-zero'
        assert np.all(sum(spin_systematics_maps.values()).imag < 1e-14), 'The sum of the input systematics maps must be real, the imaginary part is not expected to be non-zero'

        if build_systematics_maps_from_templates and dict_utils_template_systematics_maps is not None:
            print("Template systematics maps provided, checking their consistency...", flush=True)
            assert 'template_spin_systematics_maps' in dict_utils_template_systematics_maps, 'The template systematics maps must be provided in the parameters template systematics maps with key "template_spin_systematics_maps"'
            assert 'parameters_template_systematics' in dict_utils_template_systematics_maps, 'The parameters of the template systematics maps must be provided in the parameters template systematics maps with key "parameters_template_systematics"'

            assert isinstance(dict_utils_template_systematics_maps['template_spin_systematics_maps'], Spin_maps), 'The template systematics maps must be a Spin_maps instance'
            assert np.all(sum(dict_utils_template_systematics_maps['template_spin_systematics_maps'].values()).imag < 1e-14), 'The sum of the template systematics maps must be real, the imaginary part is not expected to be non-zero'
            if dict_utils_template_systematics_maps['parameters_template_systematics'].ndim > 1:
                assert np.array([dict_utils_template_systematics_maps['template_spin_systematics_maps'][spin].shape[0] == dict_utils_template_systematics_maps['parameters_template_systematics'].shape[0] for spin in dict_utils_template_systematics_maps['template_spin_systematics_maps'].spins]).all() , 'The number of templates in the template systematics maps must be the same as the number of parameters in the template systematics maps'


        for spin in set(spin_sky_maps.spins):
            if spin not in spin_systematics_maps:
                spin_systematics_maps[spin] = np.zeros(1) # If a spin is not provided in the systematics maps, we assume that the systematics maps for this spin are zero

        for spin in set(spin_systematics_maps.spins):
            if spin not in spin_sky_maps:
                spin_sky_maps[spin] = np.zeros(1) # If a spin is not provided in the sky maps, we assume that the sky maps for this spin are zero

        if projection_pixel == 'car':
            assert np.isin(
                ['shape_car', 'wcs_car'],
                list(geometry_dictionary.keys())
            ).all(), 'For CAR pixelization, the wcs and shape of the downgraded map must be provided'
            shape_final_map = geometry_dictionary['shape_car']
            wcs_car_final_map = geometry_dictionary['wcs_car']

            shape_hmap = h_n_spin_dict.shape_fullsky
            assert len(shape_hmap) == 2, 'The shape of the downgraded map must be 2D for CAR pixelization'
            wcs_car_downgraded_map = h_n_spin_dict.wcs_car

            ratio_resolution_change = (
                shape_final_map[0]//shape_hmap[0],
                shape_final_map[1]//shape_hmap[1]
            )
            assert ratio_resolution_change[0] >= 1 and ratio_resolution_change[1] >= 1, 'The resolution change can only be performed from a lower resolution to a higher resolution, the provided shape of the final map should be higher than the shape of the downgraded map'
            perform_resolution_change = True if ratio_resolution_change != (1,1) else False
            if perform_resolution_change:
                print(f"Performing resolution change from {shape_hmap} to {shape_final_map} with ratio {ratio_resolution_change}...", flush=True)
                # Handling the mask for the downgraded and final maps
                mask_downgraded = mask.reshape(shape_final_map).downgrade(factor=ratio_resolution_change)
                mask_downgraded[mask_downgraded != 0] = 1

        elif projection_pixel == 'healpix':
            raise NotImplementedError('Healpix projection is not implemented yet.')

        # Preparing interpolation
        if perform_det_interpolation:
            assert dict_utils_interpolation_det is not None, 'The utils for the interpolation of the detector maps must be provided in the parameters with key "dict_utils_interpolation_det"'
            assert 'dict_positions_detector_known' in dict_utils_interpolation_det, 'The positions of the known detectors must be provided in the utils for the interpolation of the detector maps with key "dict_positions_detector_known"'
            assert 'dict_positions_detector_unkown' in dict_utils_interpolation_det, 'The positions of the unknown detectors must be provided in the utils for the interpolation of the detector maps with key "dict_positions_detector_unkown"'
            assert 'dict_key_positions' in dict_utils_interpolation_det, 'The keys for the positions of the detectors must be provided in the utils for the interpolation of the detector maps with key "dict_key_positions"'

            dict_positions_detector_known = dict_utils_interpolation_det['dict_positions_detector_known']
            dict_positions_detector_unkown = dict_utils_interpolation_det['dict_positions_detector_unkown']
            dict_key_positions = dict_utils_interpolation_det['dict_key_positions']

            total_number_detectors = dict_positions_detector_unkown[dict_key_positions['position_x_key']].size + dict_positions_detector_known[dict_key_positions['position_x_key']].size

            list_weights = dict_utils_interpolation_det.get('list_weights', np.ones(total_number_detectors))
            inverse_weights = 1 / list_weights[...,None] if list_weights.ndim == 1 else np.where(
                list_weights !=0,
                1/list_weights,
                0
            )
            inverse_weight_spin_0 = dict_utils_interpolation_det.get('inverse_weight_to_apply_spin_0', inverse_weights)

            inverse_weight_func = lambda spin: inverse_weight_spin_0 if spin == spin_0 else inverse_weights



        # Retrieving chunks
        list_chunks_downgraded_map = get_chunks_from_shape(shape_hmap, number_chunks=number_chunks)
        list_chunks_final_map = get_chunks_final_map_from_downgraded_chunks(
            shape_final_map=shape_final_map,
            shape_map_downgraded=shape_hmap,
            list_chunks_previous_map=list_chunks_downgraded_map
        )


        print("Finishing the mapmaking process, computing the total maps...", flush=True)
        # Second, form the data vector composed of (<d_j>, <d_j cos 2\phi_j>, <d_j sin 2\phi_j>)



        final_CMB_fields = np.zeros((len(self.list_spin_output),npix), dtype=complex)

        if inverse_mapmaking_matrix is None:
            final_inverse_mapmaking_matrix = np.zeros((npix, self.nstokes, self.nstokes), dtype=complex)
            rebuild_mapmaking_matrix = True
        else:
            assert inverse_mapmaking_matrix.shape == (npix, self.nstokes, self.nstokes), 'The inverse mapmaking matrix must be of shape (npix, nstokes, nstokes), with npix being the number of pixels in the observed area of the provided mask'
            final_inverse_mapmaking_matrix = inverse_mapmaking_matrix
            rebuild_mapmaking_matrix = False

        if perform_resolution_change:
            mask_reupgraded = mask_downgraded.upgrade(factor=ratio_resolution_change).ravel()
            mask_reupgraded[mask_reupgraded != 0] = 1
            projector_map_resolution_chunk = get_projector_map_resolution(
                projection_pixel=projection_pixel,
                wcs_downgraded=wcs_car_downgraded_map,
                shape_map_downgraded=shape_hmap,
                factor=ratio_resolution_change,
                boolean_mask=mask_downgraded.ravel() != 0,
            )[mask_reupgraded != 0]


        for j, chunk_elem in enumerate(list_chunks_downgraded_map):

            print(f"Processing chunk {chunk_elem}...", flush=True)

            #TODO: Add comments everywhere here
            slice_x_downgraded = slice(chunk_elem[0], chunk_elem[1])
            slice_y_downgraded = ... if chunk_elem.size == 2 else slice(chunk_elem[2], chunk_elem[3])

            slice_final_map_x = slice(list_chunks_final_map[j][0], list_chunks_final_map[j][1])
            slice_final_map_y = ... if list_chunks_final_map[j].size == 2 else slice(list_chunks_final_map[j][2], list_chunks_final_map[j][3])

            dict_info_map_downgraded = {'shape_map':shape_hmap, 'slice_x':slice_x_downgraded, 'slice_y':slice_y_downgraded}
            dict_info_map_final = {'shape_map':shape_final_map, 'slice_x':slice_final_map_x, 'slice_y':slice_final_map_y}




            slice_1d_final_map = get_slice_1d_from_2d(
                **dict_info_map_final,
                boolean_mask=mask != 0
            )


            if perform_resolution_change:
                slice_1d_hmap = get_slice_1d_from_2d(
                    **dict_info_map_downgraded,
                    boolean_mask=mask_downgraded != 0
                )

                slice_1d_final_map_reupgraded = get_slice_1d_from_2d(
                    **dict_info_map_final,
                    boolean_mask=mask_reupgraded != 0
                )

                common_indices_slice = mask.reshape(shape_final_map)[
                    slice_final_map_x, slice_final_map_y
                ][mask_reupgraded.reshape(shape_final_map)[
                    slice_final_map_x, slice_final_map_y
                ]!=0] !=0

                slice_hmaps_from_fullsky = ...,(projector_map_resolution_chunk[slice_1d_final_map_reupgraded])[common_indices_slice]

                mask_downgraded_sliced = deepcopy(mask_downgraded)
                mask_downgraded_sliced[:] = 0
                mask_downgraded_sliced[slice_x_downgraded, slice_y_downgraded] = 1
                mask_downgraded_sliced[:] = mask_downgraded_sliced[:] * mask_downgraded[:]

                mask_reupgraded_sliced = mask_downgraded_sliced.upgrade(factor=ratio_resolution_change).ravel()
                mask_reupgraded_sliced[mask_reupgraded_sliced != 0] = 1


                common_indices_slice_v2 = mask.reshape(shape_final_map)[
                        slice_final_map_x, slice_final_map_y
                    ][mask_reupgraded_sliced.reshape(shape_final_map)[
                        slice_final_map_x, slice_final_map_y
                    ]!=0] !=0

                # hmaps
                slice_hmaps_from_downgraded_slice = (get_projector_map_resolution(
                        projection_pixel=projection_pixel,
                        wcs_downgraded=wcs_car_downgraded_map,
                        shape_map_downgraded=shape_hmap,
                        factor=ratio_resolution_change,
                        boolean_mask=mask_downgraded_sliced.ravel() != 0,
                    )[
                    mask_reupgraded_sliced.ravel() != 0
                ])[common_indices_slice_v2]


            else:
                # In this case perform_resolution_change is False
                # and slice_1d_hmap = slice_1d_final_map
                # so we can directly use it for the projection of the h_n maps

                slice_1d_hmap = slice_1d_final_map

                slice_hmaps_from_fullsky = ...,slice_1d_final_map

                slice_hmaps_from_downgraded_slice = ...

            slice_spin_sky_maps = slice_1d_final_map if spin_sky_maps[list(spin_sky_maps.spins)[0]].size != 1 else ...


            if perform_det_interpolation:
                print("Performing interpolation of the h_n maps for the current chunk...", flush=True)

                list_spin_positive = [spin for spin in h_n_spin_dict.spins if spin >= 0]

                # Different slices are taken depending on wether a resolution change is applied or not
                slice_output_interpolation = slice_hmaps_from_downgraded_slice if perform_resolution_change else None

                # If perform_det_interpolation and perform_resolution_change are both True,
                # then the output of the interpolation will be directly projected to the final map resolution
                slice_hmaps_from_fullsky = ...

                spin_non_0 = [spin for spin in h_n_spin_dict.spins if spin != spin_0][0]
                slice_weight = ...,slice_1d_hmap if inverse_weight_func(spin_non_0).shape[1] != 1 else ...

                h_n_spin_to_use = reweight_hmaps_by_hits(
                    perform_interpolation_scipy(
                        dictionary_detector_to_interpolate=dict_positions_detector_unkown,
                        dictionary_known_values={
                            **dict_positions_detector_known,
                            **{spin: h_n_spin_dict[spin][...,slice_1d_hmap] * inverse_weight_func(spin)[slice_weight] for spin in list_spin_positive}
                        },
                        method_interpolation='linear',
                        return_as_spin_maps=True,
                        slice_output=slice_output_interpolation,
                        **dict_key_positions,
                        stack_output_with_known_values=True,
                    ),
                    list_weights = np.ones(total_number_detectors),
                    list_spin=list_spin_positive
                )
                for spin in list_spin_positive:
                    h_n_spin_to_use[-spin] = np.conj(h_n_spin_to_use[spin])

                slice_for_inverse_mapmaking_matrix = None

            else:
                h_n_spin_to_use = h_n_spin_dict
                slice_for_inverse_mapmaking_matrix = slice_1d_hmap

            if inverse_mapmaking_matrix is None:
                final_inverse_mapmaking_matrix_sliced = self.get_inverse_mapmaking_matrix(
                    h_n_spin_to_use,
                    slice_1d_hmap.size,
                    polar_angle_coeff=polar_angle_coeff,
                    slice_to_apply=slice_for_inverse_mapmaking_matrix
                )[slice_hmaps_from_downgraded_slice]
            else:
                final_inverse_mapmaking_matrix_sliced = final_inverse_mapmaking_matrix[slice_1d_final_map]

            def build_on_the_fly_spin_systematics_maps(spin):
                slice_spin_systematics_maps = (...,slice_1d_final_map) if spin_systematics_maps[list(spin_systematics_maps.spins)[0]].size != 1 else ...
                if build_systematics_maps_from_templates:
                    return contract(
                        '...d,...p->dp',
                        dict_utils_template_systematics_maps['parameters_template_systematics'],
                        dict_utils_template_systematics_maps['template_spin_systematics_maps'][spin][slice_spin_sky_maps]
                    ) + spin_systematics_maps[spin][slice_spin_systematics_maps]
                else:
                    return spin_systematics_maps[spin][slice_spin_systematics_maps]


            print("Computing the spin coupled maps...", flush=True)
            spin_coupled_maps = np.zeros((slice_1d_final_map.size, len(self.list_spin_output),), dtype=complex)
            list_spin_maps = list(np.unique(spin_sky_maps.spins + spin_systematics_maps.spins))

            factor_func = lambda spin: 1 if np.sum(spin) == 0 else .5
            # Depends on the definition of the pointing matrix

            for i, spin in enumerate(self.list_spin_input):
                # Get all combinations of spins (k-k', k') such that k-k' = spin
                coupled_spins = get_coupled_spin(
                    reference_spin=spin,
                    available_h_n_spin=h_n_spin_dict.spins,
                    available_signal_spins=list_spin_maps
                )

                print(f'Coupled spins for spin {spin}: {coupled_spins}', flush=True)

                # \sum_{k' = -\infty}^{\infty} h_{k-k'} S_{k'} on all (k-k', k') pairs
                for tuple_spins in coupled_spins:
                    spin_coupled_maps[...,i] += factor_func(spin) * contract(
                        'd,d...,d...->...',
                        polar_angle_coeff[spin],
                        h_n_spin_to_use[tuple_spins[0]][slice_hmaps_from_fullsky],
                        build_on_the_fly_spin_systematics_maps(tuple_spins[1])
                        + contract(
                        'd,...->d...',
                        polar_angle_coeff[-tuple_spins[1]],
                        spin_sky_maps[tuple_spins[1]][slice_spin_sky_maps],
                    )
                ) # Polarization angle to remove for consistent modeling of the data

            get_final_map = lambda x: x if wcs_car_final_map is None else enmap.ndmap(
                x,
                wcs=wcs_car_final_map
            )

            print("Computing the final CMB fields...", flush=True)
            # Finally, compute the final CMB fields
            final_CMB_fields[...,slice_1d_final_map] = contract(
                        '...ij,...j->i...',
                        final_inverse_mapmaking_matrix_sliced,
                        spin_coupled_maps,
                    )

            if rebuild_mapmaking_matrix:
                final_inverse_mapmaking_matrix[slice_1d_final_map] = final_inverse_mapmaking_matrix_sliced

        final_CMB_fields = get_final_map(final_CMB_fields)

        print("Final CMB fields computed, transforming them into Spin_maps...", flush=True)
        dict_final_CMB_fields = Spin_maps.from_list_maps(
            final_CMB_fields,
            self.list_spin_output
        )

        if return_Q_U:
            final_Q = (dict_final_CMB_fields[-2] + dict_final_CMB_fields[2])/2.
            final_U = 1j*(dict_final_CMB_fields[-2] - dict_final_CMB_fields[2])/2.
            if self.nstokes == 3:
                final_I = dict_final_CMB_fields[0]
                output = np.vstack([final_I, final_Q, final_U])
            else:
                output = np.vstack([final_Q, final_U])
        else:
            output = dict_final_CMB_fields

        if return_inverse_mapmaking_matrix:
            return output, final_inverse_mapmaking_matrix
        return output
