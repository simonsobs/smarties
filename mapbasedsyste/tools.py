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

    minimum_spin = np.min(available_h_n_spin + available_signal_spins)
    maximum_spin = np.max(available_h_n_spin + available_signal_spins)

    # all_spins = np.arange(minimum_spin, maximum_spin+1)
    # h_n_spins = np.array(available_h_n_spin)
    # signal_spins = np.array(available_signal_spins)
    # contribution_h_n_spin = np.isin(all_spins, h_n_spins)
    # contribution_signal_spin = np.isin(all_spins, signal_spins)

    coupled_spin = []
    for spin in range(minimum_spin, maximum_spin+1):
        if spin not in available_h_n_spin:
            continue
        if reference_spin - spin in available_signal_spins and spin in available_signal_spins:
            coupled_spin.append((reference_spin - spin, spin))
    return coupled_spin
