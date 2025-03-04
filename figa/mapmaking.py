import numpy as np

from figa.tools import get_coupled_spin
from figa.cmb import create_CMB_spin_maps

class SystematicsSimulation(object):
    """
    Class to simulate systematics maps
    """

    def __init__(self, nside, nstokes, lmax, list_spin_output=[-2,2]):
        """
        Initialize the SystematicsSimulation class allowing to simulate systematics maps

        Parameters
        ----------
        nside: int
            nside of the maps
        nstokes: int
            number of Stokes parameters : 1 for the intensity only, 2 for the polarization only and 3 for the full Stokes parameters (T, Q, U)
        lmax: int
            maximum multipole (useful for CMB and systematics generation)
        list_spin: list[int]
            list of spins involved in the signal maps only (for CMB only, the spins are -2, 2 with polarization only)
        """
        self.nside = nside
        assert np.unique(list_spin_output).size == np.array(list_spin_output).size, 'The list of spins must be unique'
        if not np.isin(list_spin_output, np.array([0,-2,2])).all():
            raise NotImplemented('The spins involved in the signal maps must be 0, -2 or 2, other spins are not implemented yet')
        if nstokes > 1:
            assert list_spin_output[-2:] == [-2, 2], 'The spins involved in the signal maps must be ordered as -2, 2 for the polarization part'
        self.list_spin_output = list_spin_output # list spins involved in the signal maps only (for CMB only, the spins are 0, -2, 2 if intensity is involved)
        self.nstokes = nstokes
        if nstokes == 1:
            raise NotImplemented('The intensity only case is not implemented yet')
        self.lmax = lmax
        
    @property
    def npix(self):
        return 12*self.nside**2

    def get_spin_cmb_maps(self, seed=42):
        """
        Get the spin CMB maps which are the following for intensity and polarization:
            * Spin 0: I
            * Spin 2: Q + iU
            * Spin -2: Q - iU
        
        Parameters
        ----------
        seed: int
            seed for the random generation of the CMB maps

        Returns
        -------
        spin_CMB_maps: dictionary of spin CMB maps
            dictionary of spin CMB maps, each of shape (n_spin, npix), with n_spin being 1 if nstokes=1 (spin=0), 2 if nstokes=2 (spin=-2, 2) and 3 if nstokes=3 (spin=0, -2, 2) 

        Note
        ----
        The ordering of the spins must be the given as [0,-2,2] or [-2,2], depending on the number of Stokes parameters,
        currently the other cases are not implemented for the input spin maps
        """
        return create_CMB_spin_maps(self.nside, self.nstokes, self.lmax, seed=seed)

    def compute_total_maps(self, mask, h_n_spin_dict, spin_CMB_maps, spin_systematics_maps, return_Q_U=False):
        """
        Compute the total maps from the $h_n$ maps, the spin CMB maps and the spin systematics maps

        Parameters
        ----------
        mask: np.ndarray
            mask of the maps, the
        h_n_spin_dict: dict
            dictionary of the summed $h_n$ maps, with the keys being the spins and the values the $h_n$ maps
        spin_CMB_maps: dict
            dictionary of the spin CMB maps, with the keys being the spins and the values the spin CMB maps (e.g. if nstokes=3, the keys are 0, -2, 2 and the fields (I, Q-iU, Q+iU))
        spin_systematics_maps: dict
            dictionary of the spin systematics maps, with the keys being the spins and the values the spin systematics maps
        return_Q_U: bool
            if True, return the Q and U maps instead of the spin -2 and 2 maps 
        """

        #TODO: Only perform the inversion on the observed part
        
        # First, form the mapmaking matrix composed of the h_n map
        mapmaking_matrix = np.zeros((self.npix, self.nstokes, self.nstokes), dtype=np.complex128)
        if self.nstokes == 3:
            mapmaking_matrix[:,-3,-2] = 1 # Spin 00
            mapmaking_matrix[:,-3,-1] = .5 * h_n_spin_dict[2] # Spin 20
            mapmaking_matrix[:,-3,-1] = .5 * h_n_spin_dict[-2] # Spin -20
        mapmaking_matrix[:,-2,-2] = .5 * h_n_spin_dict[4]
        mapmaking_matrix[:,-2,-1] = 1/4.
        mapmaking_matrix[:,-1,-1] = 1/4.*h_n_spin_dict[-4]
        
        mapmaking_matrix = (mapmaking_matrix + mapmaking_matrix.transpose(0,2,1))/2. # to symmetrize the matrix

        # Second, form the data vector composed of (<d_j>, <d_j cos 2\phi_j>, <d_j sin 2\phi_j>)

        # Form the total spin maps
        list_spin_maps = np.unique(list(spin_CMB_maps.keys()) + list(spin_systematics_maps.keys()))
        for spin in list_spin_maps:
            if spin not in spin_CMB_maps:
                spin_CMB_maps[spin] = 0
            if spin not in spin_systematics_maps:
                spin_systematics_maps[spin] = 0

        total_spin_maps =  {spin: spin_CMB_maps[spin] + spin_systematics_maps[spin] for spin in list_spin_maps}
        
        spin_coupled_maps = np.zeros((self.npix, len(self.list_spin_output)), dtype=complex)
        for i, spin in enumerate(self.list_spin_output):
            coupled_spins = get_coupled_spin(spin, h_n_spin_dict.keys(), list_spin_maps)
            
            # \sum_{k' = -\infty}^{\infty} h_{k-k'} S_{k'} on all (k-k', k) pairs
            for tuple_spins in coupled_spins:
                spin_coupled_maps[:,i] += .5 * h_n_spin_dict[tuple_spins[0]]*total_spin_maps[tuple_spins[1]]

        # Third, compute the inverse of the mapmaking matrix
        inverse_mapmaking_matrix = np.linalg.pinv(mapmaking_matrix)

        # Finally, compute the final CMB fields
        final_CMB_fields = np.einsum('pij,pj->pi', inverse_mapmaking_matrix, spin_coupled_maps)

        if return_Q_U:
            final_Q = (final_CMB_fields[:,-2] + final_CMB_fields[:,-1])/2.
            final_U = 1j*(final_CMB_fields[:,-2] - final_CMB_fields[:,-1])/2.
            if self.nstokes == 3:
                final_I = final_CMB_fields[:,-3]
                return np.vstack([final_I, final_Q, final_U])
            else:
                return np.vstack([final_Q, final_U])
        else:
            return final_CMB_fields
