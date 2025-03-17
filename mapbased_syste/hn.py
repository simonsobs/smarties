from collections.abc import Iterable
import numpy as np

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
