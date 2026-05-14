# This file is part of SMARTIES.
# Copyright (c) 2024-2026 bers of the Simons Simons Observatory Collaboration.
# lease refer to the LICENSE file in the root of this repository.


import numpy as np
from opt_einsum import contract
from pixell import enmap

from smarties.tools import get_coupled_spin, get_row_mapmaking_matrix
from smarties.sky.cmb import create_CMB_spin_maps
from smarties.hn import Spin_maps, Spin_nm

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
            mask: np.ndarray = None,
            mask_input: bool = False,
            polar_angle_coeff: np.ndarray = None
        ):
        """
        Compute the inverse of the mapmaking matrix from the h_n maps
        
        Parameters
        ----------
        h_n_spin_dict: dict or Spin_maps
            Dictionary of the summed $h_n$ maps, with the keys being the spins and the values the $h_n$ maps
        mask: np.ndarray
            Mask of the maps, only the pixels in the observed area will be considered for the inversion, default is None, then all the pixels are considered
        mask_input: bool
            If True, the input $h_n$ maps will be copied and masked, otherwise the input $h_n$ maps will not be masked and assumed to be provided in the right format. Default is False.

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
        if mask is not None:
            observed_pixels_array = mask != 0
            if mask_input:
                h_n_spin_dict = Spin_maps.from_dictionary(
                    {spin: h_n_spin_dict[spin][...,observed_pixels_array] 
                     if np.size(h_n_spin_dict[spin][0,...]) == mask.size 
                     else h_n_spin_dict[spin] 
                     for spin in h_n_spin_dict.keys()}
                    )
            
            npix = mask[observed_pixels_array].size
        else:    
            npix = h_n_spin_dict[list_spin[list_spin != 0][0]].shape[-1]
        
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
                polar_angle_coeff=polar_angle_coeff
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
            mask_input: bool = True,
            polar_angle: np.ndarray = None,
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
                mask=mask, 
                mask_input=mask_input,
                polar_angle_coeff=polar_angle_coeff
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
