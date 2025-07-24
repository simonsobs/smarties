# This file is part of SMARTIES.
# Copyright (C) 2024 CNRS / SciPol developers
#
# SMARTIES is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SMARTIES is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SMARTIES. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from opt_einsum import contract

from smarties.tools import get_coupled_spin, get_row_mapmaking_matrix
from smarties.sky.cmb import create_CMB_spin_maps
from smarties.hn import Spin_maps
class FrameworkSystematics(object):
    """
    Class to simulate systematics maps
    """

    def __init__(self, nside, nstokes, lmax, list_spin_output=[-2,2]):
        """
        Initialize the FrameworkSystematics class allowing to simulate systematics maps

        Parameters
        ----------
        nside: int
            Nside of the maps
        nstokes: int
            Number of Stokes parameters : 1 for the intensity only, 2 for the polarization only and 3 for the full Stokes parameters (T, Q, U)
        lmax: int
            Maximum multipole (useful for CMB and systematics generation)
        list_spin: list[int]
            List of spins involved in the signal maps only (for CMB only, the spins are -2, 2 with polarization only)
        """
        self.nside = nside
        assert np.unique(list_spin_output).size == np.array(list_spin_output).size, 'The list of spins must be unique'
        if not np.isin(list_spin_output, np.array([0,-2,2])).all():
            print('The output spins maps appeared to contain spin different than 0, -2 or 2, the package has not been tested in this case!', flush=True)
        self.list_spin_output = list_spin_output # list spins involved in the signal maps only (for CMB only, the spins are 0, -2, 2 if intensity is involved)
        self.nstokes = nstokes
        self.lmax = lmax
        self.npix = 12*nside**2

    @property
    def list_spin_input(self):
        if self.nstokes == 3:
            return [0, 2, -2]
        elif self.nstokes == 2:
            return [2, -2]
        else:
            raise NotImplemented('The number of Stokes parameters must be 2 (polarization only) or 3 (intensity and polarization), other cases are not implemented yet')

    def get_spin_sky_maps(self, fwhm=0., seed=42):
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
            nside=self.nside, 
            nstokes=self.nstokes, 
            lmax=self.lmax, 
            fwhm=fwhm,
            seed=seed)

    def get_inverse_mapmaking_matrix(
            self, 
            h_n_spin_dict: dict | Spin_maps,
            mask: np.ndarray = None,
            mask_input: bool = False,
            dtype: type = np.complex128
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
        dtype: type
            Data type of the output inverse mapmaking matrix, for memory efficiency, default is np.complex128.

        Returns
        -------
        inverse_mapmaking_matrix: np.ndarray
            The inverse of the mapmaking matrix, with the shape (npix, nstokes, nstokes), with npix being the number of pixels in the observed area of the provided mask
        
        Note
        ----
        This function assumes that all the necessary spins are provided in the h_n maps
        and that the h_n maps are normalized 
        """
        if mask is not None:
            assert mask.size == 12 * self.nside **2, 'The mask must be a HEALPix map of the same size as the h_n maps'
            observed_pixels_array = mask != 0
            if mask_input:
                h_n_spin_dict = Spin_maps.from_dictionary({spin: h_n_spin_dict[spin][...,observed_pixels_array] if np.size(h_n_spin_dict[spin][0,...]) == mask.size else h_n_spin_dict[spin] for spin in h_n_spin_dict.keys()})
            
            npix = mask[observed_pixels_array].size
        else:
            list_spin = np.array(list(h_n_spin_dict.keys()))
            npix = h_n_spin_dict[list_spin[list_spin != 0][0]].shape[-1]
    
        # First, form the mapmaking matrix composed of the h_n map
        mapmaking_matrix = np.zeros((npix, self.nstokes, self.nstokes), dtype=dtype)
        for i, reference_spin in enumerate(self.list_spin_output):
            mapmaking_matrix[:,i,:] = get_row_mapmaking_matrix(reference_spin, h_n_spin_dict, self.list_spin_input)
        # Then, compute the inverse of the mapmaking matrix
        return np.linalg.pinv(mapmaking_matrix)

    def compute_total_maps(
            self, 
            mask: np.ndarray, 
            h_n_spin_dict: dict | Spin_maps, 
            spin_sky_maps: dict | Spin_maps, 
            spin_systematics_maps: dict | Spin_maps, 
            inverse_mapmaking_matrix : np.ndarray = None,
            return_Q_U: bool = False,
            return_inverse_mapmaking_matrix: bool = False,
            mask_input: bool = True,
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
            Dictionary of the spin CMB maps, with the keys being the spins and the values the spin CMB maps (e.g. if nstokes=3, the keys are 0, -2, 2 and the fields (I, Q-iU, Q+iU))
        spin_systematics_maps: dict
            Dictionary of the spin systematics maps, with the keys being the spins and the values the spin systematics maps
        inverse_mapmaking_matrix : np.ndarray, optional
            The inverse of the mapmaking matrix, with the shape (npix, n_spin, n_spin), with npix being the number of pixels in the observed area of the provided mask. If None (default), it will be computed from the h_n maps
        return_Q_U: bool
            If True, return the Q and U maps instead of the spin -2 and 2 maps, default is False
        return_inverse_mapmaking_matrix: bool
            If True, return the inverse of the mapmaking matrix, default is False
        mask_input: bool
            If True, the input spin_sky_maps, spin_systematics_maps and h_n_spin_dict will be copied and masked, otherwise the input maps will not be masked and assumed to be provided in the right format. Default is True.

        Returns
        -------
        final_CMB_fields: np.ndarray
            the final CMB fields, with the shape (npix, nstokes) if return_Q_U is False, (npix, 3) otherwise
        """

        # Few assert
        assert np.allclose([spin_systematics_maps[spin].ndim for spin in spin_systematics_maps.keys() if spin != 0 ], 2), 'The systematics maps must be 2D arrays of shape (n_det, n_pix)'
        assert np.allclose([spin_sky_maps[spin].ndim for spin in spin_sky_maps.keys()], 1), 'The CMB maps must be 1D arrays of shape (n_pix)'
        assert np.allclose([h_n_spin_dict[spin].ndim for spin in h_n_spin_dict.keys() if spin != 0 ], 2), 'The h_n maps must be 2D arrays of shape (n_det, n_pix)'

        assert h_n_spin_dict[0].sum() == 1, 'The h_n maps must be normalized'

        if mask is None:
            mask = np.ones(12 * self.nside ** 2, dtype=np.int8)
        
        # Masking the h_n maps, CMB maps and systematics maps
        observed_pixels_array = mask != 0
        if mask_input:
            h_n_spin_dict = Spin_maps.from_dictionary({spin: h_n_spin_dict[spin][...,observed_pixels_array] if np.size(h_n_spin_dict[spin][0,...]) == mask.size else h_n_spin_dict[spin] for spin in h_n_spin_dict.keys()})
            spin_sky_maps = Spin_maps.from_dictionary({spin: spin_sky_maps[spin][...,observed_pixels_array] if np.size(spin_sky_maps[spin]) == mask.size else spin_sky_maps[spin] for spin in spin_sky_maps.keys()})
            spin_systematics_maps = Spin_maps.from_dictionary({spin: spin_systematics_maps[spin][...,observed_pixels_array] if np.size(spin_systematics_maps[spin][0,...]) == mask.size else spin_systematics_maps[spin] for spin in spin_systematics_maps.keys()})
            
        # else: 
        #     spin_sky_maps = Spin_maps.from_dictionary(spin_sky_maps)
        
    
        assert np.all(sum(spin_sky_maps.values()).imag < 1e-14), 'The sum of the input sky maps must be real, the imaginary part is not expected to be non-zero'
        assert np.all(sum(spin_systematics_maps.values()).imag < 1e-14), 'The sum of the input systematics maps must be real, the imaginary part is not expected to be non-zero'

        total_spin_maps = Spin_maps.from_dictionary(spin_systematics_maps) # Initialize the total spin maps with the input systematics maps

        npix = mask[observed_pixels_array].size

        if inverse_mapmaking_matrix is None:
            inverse_mapmaking_matrix = self.get_inverse_mapmaking_matrix(h_n_spin_dict, mask=mask, mask_input=False)
            # The h_n_spin_dict is not masked here, as it is assumed to be already in the right format
        else: 
            assert inverse_mapmaking_matrix.shape == (npix, self.nstokes, self.nstokes), 'The inverse mapmaking matrix must be of shape (npix, nstokes, nstokes), with npix being the number of pixels in the observed area of the provided mask'

        print("Finishing the mapmaking process, computing the total maps...", flush=True)        
        # Second, form the data vector composed of (<d_j>, <d_j cos 2\phi_j>, <d_j sin 2\phi_j>)

        # Form the total spin maps
        n_det = h_n_spin_dict[0].shape[0]
        # spin_sky_maps.extend_first_dimension(n_det) # Extend the first dimension of the spin_sky_maps to match the number of detectors before summing them to the systematics maps

        print("Summing the spin sky maps and systematics maps...", flush=True)
        total_spin_maps.add_inplace(spin_sky_maps)
        
        print("Computing the spin coupled maps...", flush=True)
        spin_coupled_maps = np.zeros((npix, len(self.list_spin_output),), dtype=complex)
        list_spin_maps = total_spin_maps.spins
        factor_dict = {0: 1, -2: .5, 2: .5}
        for i, spin in enumerate(self.list_spin_input):
            # Get all combinations of spins (k-k', k') such that k-k' = spin
            coupled_spins = get_coupled_spin(spin, h_n_spin_dict.spins, list_spin_maps)

            # TODO: Remove print
            print(f'Coupled spins for spin {spin}: {coupled_spins}', flush=True)
            
            # \sum_{k' = -\infty}^{\infty} h_{k-k'} S_{k'} on all (k-k', k') pairs
            for tuple_spins in coupled_spins:
                spin_coupled_maps[...,i] += factor_dict[spin] * contract('d...,d...->...',h_n_spin_dict[tuple_spins[0]], total_spin_maps[tuple_spins[1]])
                # spin_coupled_maps[...,i] += factor_dict[spin] * contract('dp,dp->p', h_n_spin_dict[tuple_spins[0]], total_spin_maps[tuple_spins[1]], memory_limit='max_input')
                
                # spin_coupled_maps[...,i] += factor_dict[spin] * np.einsum('d...,d...->...',h_n_spin_dict[tuple_spins[0]], total_spin_maps[tuple_spins[1]])

                # spin_coupled_maps[...,i] += factor_dict[spin] * (h_n_spin_dict[tuple_spins[0]] * total_spin_maps[tuple_spins[1]]).sum(axis=0)

        del total_spin_maps
        print("Computing the final CMB fields...", flush=True)
        # Finally, compute the final CMB fields
        final_CMB_fields = contract('pij,pj->ip', inverse_mapmaking_matrix, spin_coupled_maps) #, memory_limit='max_input') ## Memory handling necessary?
        # final_CMB_fields = np.einsum('pij,pj->ip', inverse_mapmaking_matrix, spin_coupled_maps)

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

    def compute_total_maps_v2(
            self, 
            mask: np.ndarray, 
            h_n_spin_dict: dict | Spin_maps, 
            spin_sky_maps: dict | Spin_maps, 
            spin_systematics_maps: dict | Spin_maps, 
            inverse_mapmaking_matrix : np.ndarray = None,
            return_Q_U: bool = False,
            return_inverse_mapmaking_matrix: bool = False,
            mask_input: bool = True,
            polar_angle: bool = None
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
            Dictionary of the spin CMB maps, with the keys being the spins and the values the spin CMB maps (e.g. if nstokes=3, the keys are 0, -2, 2 and the fields (I, Q-iU, Q+iU))
        spin_systematics_maps: dict
            Dictionary of the spin systematics maps, with the keys being the spins and the values the spin systematics maps
        return_Q_U: bool
            If True, return the Q and U maps instead of the spin -2 and 2 maps, default is False
        return_inverse_mapmaking_matrix: bool
            If True, return the inverse of the mapmaking matrix, default is False

        Returns
        -------
        final_CMB_fields: np.ndarray
            the final CMB fields, with the shape (npix, nstokes) if return_Q_U is False, (npix, 3) otherwise
        """

        # Few assert
        assert np.allclose([spin_systematics_maps[spin].ndim for spin in spin_systematics_maps.keys() if spin != 0 ], 2), 'The systematics maps must be 2D arrays of shape (n_det, n_pix)'
        assert np.allclose([spin_sky_maps[spin].ndim for spin in spin_sky_maps.keys()], 1), 'The CMB maps must be 1D arrays of shape (n_pix)'
        assert np.allclose([h_n_spin_dict[spin].ndim for spin in h_n_spin_dict.keys() if spin != 0 ], 2), 'The h_n maps must be 2D arrays of shape (n_det, n_pix)'

        assert h_n_spin_dict[0].sum() == 1, 'The h_n maps must be normalized'

        if mask is None:
            mask = np.ones(12 * self.nside ** 2, dtype=np.int8)
        
        # Masking the h_n maps, CMB maps and systematics maps
        observed_pixels_array = mask != 0
        if mask_input:
            h_n_spin_dict = Spin_maps.from_dictionary({spin: h_n_spin_dict[spin][...,observed_pixels_array] if np.size(h_n_spin_dict[spin][0,...]) == mask.size else h_n_spin_dict[spin] for spin in h_n_spin_dict.keys()})
            spin_sky_maps = Spin_maps.from_dictionary({spin: spin_sky_maps[spin][...,observed_pixels_array] if np.size(spin_sky_maps[spin]) == mask.size else spin_sky_maps[spin] for spin in spin_sky_maps.keys()})
            spin_systematics_maps = Spin_maps.from_dictionary({spin: spin_systematics_maps[spin][...,observed_pixels_array] if np.size(spin_systematics_maps[spin][0,...]) == mask.size else spin_systematics_maps[spin] for spin in spin_systematics_maps.keys()})
            # TODO: Decide if input maps are masked here, or if the user should provide the masked maps directly
        # else: 
        #     spin_sky_maps = Spin_maps.from_dictionary(spin_sky_maps)
        
    
        assert np.all(sum(spin_sky_maps.values()).imag < 1e-14), 'The sum of the input sky maps must be real, the imaginary part is not expected to be non-zero'
        assert np.all(sum(spin_systematics_maps.values()).imag < 1e-14), 'The sum of the input systematics maps must be real, the imaginary part is not expected to be non-zero'

        total_spin_maps = Spin_maps.from_dictionary(spin_systematics_maps) # Initialize the total spin maps with the input systematics maps

        npix = mask[observed_pixels_array].size

        if inverse_mapmaking_matrix is None:
            inverse_mapmaking_matrix = self.get_inverse_mapmaking_matrix(h_n_spin_dict, mask=mask, mask_input=mask_input)
        else: 
            assert inverse_mapmaking_matrix.shape == (npix, self.nstokes, self.nstokes), 'The inverse mapmaking matrix must be of shape (npix, nstokes, nstokes), with npix being the number of pixels in the observed area of the provided mask'

        print("Finishing the mapmaking process, computing the total maps...", flush=True)        
        # Second, form the data vector composed of (<d_j>, <d_j cos 2\phi_j>, <d_j sin 2\phi_j>)

        # Form the total spin maps
        n_det = h_n_spin_dict[0].shape[0]
        # spin_sky_maps.extend_first_dimension(n_det) # Extend the first dimension of the spin_sky_maps to match the number of detectors before summing them to the systematics maps

        if polar_angle is None:
            polar_angle_coeff = {spin: np.ones(h_n_spin_dict[spin].shape[0], dtype=complex) for spin in h_n_spin_dict.spins}
        else:
            assert polar_angle.shape == h_n_spin_dict[0].shape, 'The polar angle map must have the same shape as the h_n maps'
            polar_angle_coeff = {spin: np.exp(spin * 1j * polar_angle) for spin in h_n_spin_dict.spins}

        print("Summing the spin sky maps and systematics maps...", flush=True)
        # total_spin_maps.add_inplace(spin_sky_maps)
        
        print("Computing the spin coupled maps...", flush=True)
        spin_coupled_maps = np.zeros((npix, len(self.list_spin_output),), dtype=complex)
        list_spin_maps = spin_sky_maps.spins #total_spin_maps.spins
        factor_dict = {0: 1, -2: .5, 2: .5}
        for i, spin in enumerate(self.list_spin_input):
            # Get all combinations of spins (k-k', k') such that k-k' = spin
            coupled_spins = get_coupled_spin(spin, h_n_spin_dict.spins, list_spin_maps)

            # TODO: Remove print
            print(f'Coupled spins for spin {spin}: {coupled_spins}', flush=True)
            
            # \sum_{k' = -\infty}^{\infty} h_{k-k'} S_{k'} on all (k-k', k') pairs
            for tuple_spins in coupled_spins:
                spin_coupled_maps[...,i] += factor_dict[spin] * contract('d,d...,d...->...', polar_angle_coeff[-tuple_spins[0]], h_n_spin_dict[tuple_spins[0]], spin_systematics_maps[tuple_spins[1]]) + factor_dict[spin] * contract('dp,p->p',h_n_spin_dict[tuple_spins[0]], spin_sky_maps[tuple_spins[1]])
                # spin_coupled_maps[...,i] += factor_dict[spin] * contract('dp,dp->p', h_n_spin_dict[tuple_spins[0]], total_spin_maps[tuple_spins[1]], memory_limit='max_input')
                
                # spin_coupled_maps[...,i] += factor_dict[spin] * np.einsum('d...,d...->...',h_n_spin_dict[tuple_spins[0]], total_spin_maps[tuple_spins[1]])

                # spin_coupled_maps[...,i] += factor_dict[spin] * (h_n_spin_dict[tuple_spins[0]] * total_spin_maps[tuple_spins[1]]).sum(axis=0)

        print("Computing the final CMB fields...", flush=True)
        # Finally, compute the final CMB fields
        final_CMB_fields = contract('pij,pj->ip', inverse_mapmaking_matrix, spin_coupled_maps) #, memory_limit='max_input') ## Memory handling necessary?
        # final_CMB_fields = np.einsum('pij,pj->ip', inverse_mapmaking_matrix, spin_coupled_maps)

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
