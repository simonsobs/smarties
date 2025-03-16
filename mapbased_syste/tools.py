import numpy as np


def get_coupled_spin(reference_spin, available_h_n_spin, available_signal_spins):
    """
    Get the coupled spins for a reference spin $k$ given a set of $h_n$ and signal spin maps, involved in a typical sum: 
        $ \sum_{k' = -\infty}^{\infty} h_{k-k'} S_{k'}$
        
    Parameters
    ----------
    reference_spin: int
        reference spin $k$
    available_h_n_spin: list[int]
        list of available $h_n$ spins, typically [-4, -2, 2, 4]
    available_signal_spins: list[int]
        list of available signal spins, typically [-2, 2]
    
    Returns
    -------
    coupled_spin: list[tuple]
        list of available coupled spins, each tuple being the coupled spins $(k-k', k')$
    """

    minimum_spin = np.min(list(available_h_n_spin) + list(available_signal_spins))
    maximum_spin = np.max(list(available_h_n_spin) + list(available_signal_spins))

    # all_spins = np.arange(minimum_spin, maximum_spin+1)
    # h_n_spins = np.array(available_h_n_spin)
    # signal_spins = np.array(available_signal_spins)
    # contribution_h_n_spin = np.isin(all_spins, h_n_spins)
    # contribution_signal_spin = np.isin(all_spins, signal_spins)

    coupled_spin = []
    for spin in range(minimum_spin, maximum_spin+1):
        if reference_spin - spin in available_h_n_spin and spin in available_signal_spins:
            coupled_spin.append((reference_spin - spin, spin))
    return coupled_spin


def get_row_mapmaking_matrix(reference_spin, h_n_spin_dict, list_spin_input):
    """
    The mapmaking matrix will always be multiplied to the vector ordered with spins [0, 2, -2] for the $\tilde{S}^{\rm pixel}_{k}$ term.

    The $h_n$ dictionary must have the following structure:
        h_n_spin_dict = {spin: np.array([n_det, n_pix])} (for spin != 0)
        h_n_spin_dict = {0: np.array([1, 1])} (for spin = 0)
    And the $h_n$ maps will be summed over the detectors in the mapmaking matrix, so that the mapmaking matrix will be of shape [n_spin, n_pix] with n_spin the number of spins involved in list_spin_input and n_pix the number of pixels in the $h_n$ maps.
    """

    factor_func = lambda x: 1 if x == 0 else .5

    mapmaking_matrix_row = np.zeros((h_n_spin_dict[2].shape[-1], len(list_spin_input)), dtype=complex)
    for i, spin_name in enumerate(list_spin_input):
        mapmaking_matrix_row[:,i] = factor_func(reference_spin) * factor_func(spin_name) * h_n_spin_dict[spin_name-reference_spin].sum(axis=0)
        
    return mapmaking_matrix_row
