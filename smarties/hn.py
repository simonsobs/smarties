# This file is part of SMARTIES.
# Copyright (C) 2024-2026 Members of the Simons Observatory collaboration.
# Please refer to the LICENSE file in the root of this repository.


from collections.abc import Iterable

import healpy as hp
import numpy as np
from opt_einsum import contract
from pixell import enmap

__all__ = [
    'Spin_nm',
    'Spin_maps'
]

class Spin_nm(tuple):
    """
    Class to handle fundamental operations related to the spin numbers.
    """
    def __neg__(self):
        return Spin_nm([-item for item in self.__iter__()])

    def __add__(self, other):
        return Spin_nm([item1 + item2 for item1, item2 in zip(self.__iter__(), other)])

    def __sub__(self, other):
        return Spin_nm([item1 - item2 for item1, item2 in zip(self.__iter__(), other)])


class Spin_maps(dict):
    """
    Class to handle fundamental operations related to the spin maps.

    The class can be defined from a dictionary, or from a list of maps and a list of spins. 
    The keys of the dictionary are the spins, and the values are the corresponding maps.

    WARNNING: The wcs and shape attributes as 'shape_fullsky' and 'wcs_car' are
    not set and neither updated dynamically when creating the Spin_maps object,
    but they can be set and accessed through the corresponding properties.
    """

    _projection_pixel = None
    _shape_fullsky = None
    _wcs_car = None
    _total_hits_map = None

    def __missing__(self, key):
        if (isinstance(key, int) or isinstance(key, Spin_nm) or isinstance(key, np.integer)) and -key in self: 
            return np.conj(super().__getitem__(-key))
        raise  KeyError(key)

    def set_projection_pixel(self, projection_pixel=None):
        """Set the pixelisation of the maps, ``'car'`` or ``'healpix'``."""
        assert projection_pixel in ["car", "healpix"], (
            f"Projection pixel must be either 'car' or 'healpix', got {projection_pixel}"
        )
        self._projection_pixel = projection_pixel

    def get_projection_pixel(self):
        """Pixelisation of the maps, ``'car'`` or ``'healpix'`` (None if not set)."""
        return self._projection_pixel

    projection_pixel = property(get_projection_pixel, set_projection_pixel)

    def set_shape_fullsky(self, shape=None):
        """Set the shape of the full (unmasked) map, ``(npix,)`` or ``(ny, nx)``."""
        self._shape_fullsky = shape

    def get_shape_fullsky(self):
        """Shape of the full (unmasked) map, ``(npix,)`` or ``(ny, nx)`` (None if not set)."""
        return self._shape_fullsky

    shape_fullsky = property(get_shape_fullsky, set_shape_fullsky)

    def set_wcs_car(self, wcs=None):
        """Set the WCS of the CAR maps."""
        self._wcs_car = wcs

    def get_wcs_car(self):
        """WCS of the CAR maps (None for HEALPix maps or if not set)."""
        return self._wcs_car

    wcs_car = property(get_wcs_car, set_wcs_car)

    def set_total_hits_map(self, total_hits_map=None):
        """Set the total hits per pixel, summed over the detectors."""
        self._total_hits_map = total_hits_map

    def get_total_hits_map(self):
        """Total hits per pixel, summed over the detectors, set by the $h$-map readers (None if not set)."""
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

        Parameters
        ----------
        dictionary: dict
            ``spin -> map``; the maps are not copied, nor the metadata.

        Returns
        -------
        Spin_maps
            A new Spin_maps object holding the same maps.
        """

        result = cls()
        for key, value in dictionary.items():
            result[key] = value

        return result

    @classmethod
    def from_list_maps(cls, maps, list_spin):
        """
        Create a Spin_maps object from a list of maps and a list of spins

        Parameters
        ----------
        maps: np.ndarray | enmap.ndmap | list
            Maps, the first dimension running over the spins; for an ``enmap.ndmap``, the
            maps of the result are ``enmap.ndmap`` with the same WCS.
        list_spin: Iterable
            Spin of each map.

        Returns
        -------
        Spin_maps
            ``spin -> map``.
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

        Parameters
        ----------
        other: Spin_maps
            Spin maps to add: the maps of the spins present in both are added, the others are
            added to this object.
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

        Notes
        -----
        The spin 0 maps are left unchanged.
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
        subscripts: str
            Einstein summation subscripts for the multiplication operation. Default is
            'd...,d...->d...'.

        Notes
        -----
        The spin 0 maps are left unchanged.
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

        Parameters
        ----------
        new_shape_first_dimension: int
            Size of the new first dimension, e.g. the number of detectors.

        Notes
        -----
        A broadcast is performed to add a first dimension to each element of the dictionary:
        a map of shape ``(npix,)`` becomes ``(new_shape_first_dimension, npix)``, without copy.
        """
        for key in self.keys():
            self[key] = np.broadcast_to(
                self[key], (new_shape_first_dimension,) + np.asarray(self[key]).shape
            )

    def copy_sliced(self, slice_1d):
        """
        Copy the spin maps with a slice on the last (pixel) dimension

        Parameters
        ----------
        slice_1d: slice | np.ndarray
            Slice or indices to apply on the last dimension of each element of the dictionary;
            the maps of size 1 are kept as they are. The metadata are not copied.

        Returns
        -------
        Spin_maps
            A new Spin_maps object with the sliced elements.
        """
        result = Spin_maps()
        for key in self.keys():
            if self[key].size != 1:
                result[key] = self[key][...,slice_1d]
            else:
                result[key] = self[key]
        return result