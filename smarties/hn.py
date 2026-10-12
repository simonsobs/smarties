# This file is part of SMARTIES.
# Copyright (c) 2024-2026 bers of the Simons Simons Observatory Collaboration.
# lease refer to the LICENSE file in the root of this repository.


from collections.abc import Iterable

import healpy as hp
import numpy as np
from opt_einsum import contract
from pixell import enmap

__all__ = ["Spin_nm", "Spin_maps"]


class Spin_nm(tuple):
    def __neg__(self):
        return Spin_nm([-item for item in self.__iter__()])

    def __add__(self, other):
        return Spin_nm([item1 + item2 for item1, item2 in zip(self.__iter__(), other)])

    def __sub__(self, other):
        return Spin_nm([item1 - item2 for item1, item2 in zip(self.__iter__(), other)])


class Spin_maps(dict):
    """
    Class to handle fundamental operations related to the spin maps.

    WARNNING: The wcs and shape attributes as 'shape_fullsky' and 'wcs_car' are
    not set and neither updated dynamically when creating the Spin_maps object,
    but they can be set and accessed through the corresponding properties.
    """

    _projection_pixel = None
    _shape_fullsky = None
    _wcs_car = None

    def set_projection_pixel(self, projection_pixel=None):
        assert projection_pixel in ["car", "healpix"], (
            f"Projection pixel must be either 'car' or 'healpix', got {projection_pixel}"
        )
        self._projection_pixel = projection_pixel

    def get_projection_pixel(self):
        return self._projection_pixel

    projection_pixel = property(get_projection_pixel, set_projection_pixel)

    def set_shape_fullsky(self, shape=None):
        self._shape_fullsky = shape

    def get_shape_fullsky(self):
        return self._shape_fullsky

    shape_fullsky = property(get_shape_fullsky, set_shape_fullsky)

    def set_wcs_car(self, wcs=None):
        self._wcs_car = wcs

    def get_wcs_car(self):
        return self._wcs_car

    wcs_car = property(get_wcs_car, set_wcs_car)

    def set_total_hits_map(self, total_hits_map=None):
        self._total_hits_map = total_hits_map

    def get_total_hits_map(self):
        return self._total_hits_map

    total_hits_map = property(get_total_hits_map, set_total_hits_map)

    @property
    def spins(self):
        """
        Returns the list of spins in the Spin_maps object
        """
        all_keys = self.keys() if len(self.keys()) > 1 else list(self.keys())
        define_spins_func = lambda key: (
            Spin_nm(key) if (isinstance(key, Iterable) and len(key) != 1) else key
        )
        return [define_spins_func(key) for key in all_keys]

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
        transform_into_car = lambda x: (
            x if type(maps) is not enmap.ndmap else enmap.ndmap(x, wcs=maps.wcs)
        )
        for spin, map_ in zip(list_spin, maps):
            result[spin] = transform_into_car(map_)
        return result

    def __add__(self, other):
        """
        Add two spin maps objects

        Notes
        -----
        A new object is created.
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

    def multiply_inplace_detectors_spin_maps(self, other, subscripts="d...,d...->d..."):
        """
        Multiply the spin maps by another Spin_maps object in place

        Parameters
        ----------
        other: Spin_maps
            Another Spin_maps object to multiply with.
        subscripts: str
            Einstein summation subscripts for the multiplication operation.
            Default is 'd...,d...->d...' which multiplies each `spin` map by the corresponding `spin` map of the other Spin_maps object for each detector.
        """
        assert isinstance(other, Spin_maps) or np.all(
            [key in other for key in self.keys() if key != 0]
        ), (
            "The other object must be an instance of Spin_maps or at least contain all keys of self"
        )
        assert subscripts is not None, (
            "Subscripts must be provided for the multiplication operation"
        )
        for key in self.keys():
            if key != 0:
                self[key] = contract(subscripts, self[key], other[key])

    def divide_inplace_detectors_spin_maps(self, other, subscripts="d...,d...->d..."):
        """
        Multiply the spin maps in place, with each `spin` map multiplied
        by the corresponding `-spin` map of the other Spin_maps object.

        Parameters
        ----------
        other: Spin_maps
            Another Spin_maps object to extract `-spin` maps from for multiplication.
        """
        assert isinstance(other, Spin_maps) or np.all(
            [key in other for key in self.keys() if key != 0]
        ), (
            "The other object must be an instance of Spin_maps or at least contain all keys of self"
        )
        assert subscripts is not None, (
            "Subscripts must be provided for the multiplication operation"
        )

        for key in self.keys():
            if key != 0:
                self[key] = contract(subscripts, self[key], other[-key])

    def extend_first_dimension(self, new_shape_first_dimension):
        """
        Extend the first dimension of the spin maps to a new shape

        Notes
        -----
        A broadcast is performed to extend the first dimension of each element of the dictionary.
        """
        for key in self.keys():
            self[key] = np.broadcast_to(
                self[key], (new_shape_first_dimension,) + np.asarray(self[key]).shape
            )
