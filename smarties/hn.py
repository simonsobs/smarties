from collections.abc import Iterable
import numpy as np
import healpy as hp

class Spin_maps(dict):
    """
    Class to handle the spin maps
    """
    
    @property
    def spins(self):
        return list(self.keys())

    @classmethod
    def from_dictionary(cls, dictionary):
        """
        Create a Spin_maps object from a dictionary
        """

        result = cls()
        for key, value in dictionary.items():
            result[key] = value

        return result
    
    @classmethod
    def from_list_maps(cls, maps, list_spin):
        """
        Create a Spin_maps object from a list of maps and a list of spins
        """
        assert isinstance(list_spin, Iterable)
        result = cls()
        for spin, map in zip(list_spin, maps):
            result[spin] = map
        return result
    
    def __add__(self, other):
        """
        Add two spin maps objects

        Note: a new object is created
        """

        result = Spin_maps()

        all_keys = np.unique(list(self.keys()) + list(other.keys()))
        for key in all_keys:
            if key not in self.keys():
                result[key] = other[key]
            elif key not in other.keys():
                result[key] = self[key]
            else:
                result[key] = self[key] + other[key]
        return result

    def add_inplace(self, other):
        """
        Add another Spin_maps object to this one in place
        """
        for key, value in other.items():
            if key in self:
                self[key] = self[key] + value
            else:
                self[key] = value
        
    def extend_first_dimension(self, new_shape_first_dimension):
        """
        Extend the first dimension of the spin maps to a new shape
        """
        for key in self.keys():
            self[key] = np.broadcast_to(self[key], (new_shape_first_dimension,) + np.asarray(self[key]).shape)

def ud_grade_hn(h_n_maps, nside_out):
    """
    Change the resolution of the $h_n$ maps to a lower or higher resolution, 
    by averaging or repeating the pixels in the provided output resolution 
    using the `ud_grade` function from HEALPix. 

    Parameters
    ----------
    h_n_maps: Spin_maps
        Spin maps containing the $h_n$ maps, with keys being the spins and values being
        the maps of shape (n_det, n_pix) or (n_pix,) for spin=0.
    nside_out: int
        The desired output resolution, given as nside.

    Returns
    -------
    new_h_n: Spin_maps
        A new Spin_maps object containing the $h_n$ maps at the desired resolution,
        with the same spins as the input maps. The maps are of shape (n_det, n_pix) or (n_pix,) for spin=0,
        where n_det is the number of detectors (1 for spin=0) and n_pix is the number of pixels at the output
        resolution.

    Notes
    -----
    Currently the corresponding operations only work with HEALPix maps, so the input maps must be in HEALPix format. 
    """
    
    new_h_n = Spin_maps()
    for spin in h_n_maps.spins:
        if h_n_maps[spin].ndim != 1 and h_n_maps[spin].shape[-1] != 1:
            number_of_detectors = 1 if h_n_maps[spin].ndim == 1 else h_n_maps[spin].shape[0]
            new_h_n[spin] = np.zeros((number_of_detectors, hp.nside2npix(nside_out)), dtype=h_n_maps[spin].dtype)
            for detector in range(number_of_detectors):
                new_h_n[spin][detector] = hp.ud_grade(h_n_maps[spin][detector], nside_out, power=None)
        else:
            new_h_n[spin] = h_n_maps[spin]
    return new_h_n
