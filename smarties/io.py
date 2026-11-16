# This file is part of SMARTIES.
# Copyright (C) 2024-2026 Members of the Simons Observatory collaboration.
# Please refer to the LICENSE file in the root of this repository.


import os
import logging
from copy import deepcopy
import h5py
import healpy as hp
import numpy as np
import yaml
from opt_einsum import contract
from pixell import enmap

from smarties.hn import Spin_maps, Spin_nm
from smarties.utils.tools import flatten_CAR_maps, reweight_hmaps_by_hits

logger = logging.getLogger(__name__)

FORMATS_H5 = ["h5", "hdf5", "H5", "hdf"]

__all__ = [
    "build_mask_from_toast_h_maps",
    "read_file",
    "read_toast_h_maps",
    "read_lbs_h_maps",
    "read_spherical_derivatives_from_file",
    "read_values_from_yaml_file",
    "save_partial_spin_maps",
    "read_toast_h_maps_old",
    "build_mask_from_toast_h_maps_old",
]


def _file_name(name: str, format_file: str) -> str:
    return name if name.endswith("." + format_file) else name + "." + format_file


def _read_geometry(file: h5py.File) -> dict:
    """
    Pixelisation of an h-map file: projection ('car' or 'healpix'), shape of the full map,
    WCS (CAR), and, for HEALPix, the indices converting a map in the ordering of the file to
    RING ordering (None if already in RING ordering).
    """
    attributes = file.attrs
    if str(attributes["pixelization"]).upper() == "WCS":
        header = attributes["wcs_header"]
        header = header.decode() if isinstance(header, bytes) else str(header)
        return {
            "projection_pixel": "car",
            "shape_fullsky": tuple(int(size) for size in attributes["wcs_shape"]),
            "wcs": wcsutils.WCS(fits.Header.fromstring(header)),
            "to_ring": None,
            "coordinate_system": str(attributes.get("coordinate_system", "unknown")),
        }

    nside = int(attributes["nside"])
    nested = str(attributes.get("ordering", "RING")).upper() == "NESTED"
    return {
        "projection_pixel": "healpix",
        "shape_fullsky": (hp.nside2npix(nside),),
        "wcs": None,
        "to_ring": hp.ring2nest(nside, np.arange(hp.nside2npix(nside))) if nested else None,
        "coordinate_system": str(attributes.get("coordinate_system", "unknown")),
    }


def _read_dataset(file: h5py.File, group: str, spin_nm: str, to_ring: bool) -> np.ndarray:
    """Dataset of an h-map file, flattened, in RING ordering for HEALPix."""
    if to_ring is False:
        to_ring = None
    value = file[group][spin_nm][...].ravel()
    return value if to_ring is None else value[to_ring]



def read_file(
    name_file: str,
    format_file: str = "fits",
    boolean_mask: np.ndarray = None,
    dtype=np.float64,
    projection_pixel: str = "healpix",
    flatten_output: bool = True,
) -> np.ndarray | enmap.ndmap:
    """
    Read the $h$-maps as provided either in FITS, NPY or NPZ
    format, and eventually mask them.

    Parameters
    ----------
    name_file: str
        Path to the $h$-maps file to be read. If the path provided does
        not contain the fomat suffix, in '.fits', '.npy' or '.npz',
        the suffix corresponding to the format indicated in format_file
        will be concatenated.
    format_file: str
        String indicated the format of the file to be read, must be
        either 'fits', 'npy' or 'npz' otherwise a ValueError is raised
    boolean_mask: array[bool] 
        Optional, boolean array corresponding to the selected pixels in
        the array to retain after loading the file. By default, return
        all pixels. Note that this option is not active for 'npz' format
        files. 
    dtype: type, optional
        Data type to use when reading the file. Default is np.float64.
    projection_pixel: str, optional
        Projection of the pixelization of the map to be read, either 'healpix' or 'car'. Default is 'healpix'.
    flatten_output: bool, optional
        If True, flatten the output maps for CAR projection. Default is True.

    Returns
    -------
        Array containing the $h$-maps read, from which only the relevant
        pixels are possibly retained.
    """
    if boolean_mask is None:
        boolean_mask = ...

    _flatten_CAR_maps = flatten_CAR_maps if flatten_output else lambda x: x

    name_file = _file_name(name_file, format_file)
    if format_file == "fits":
        name_file = _file_name(name_file, format_file)
        if projection_pixel == "healpix":
            return hp.read_map(name_file)[boolean_mask]
        else:
            return _flatten_CAR_maps(enmap.read_map(name_file))[boolean_mask]
    elif format_file == "npy":
        return np.memmap(name_file, mode="r", dtype=dtype)[boolean_mask]
    elif format_file == "npz":
        return np.load(name_file)  # , allow_pickle=True)
    elif format_file in FORMATS_H5:
        if projection_pixel == "healpix":
            return h5py.File(name_file, "r")
        else:
            return _flatten_CAR_maps(enmap.read_map(name_file))[boolean_mask]
    else:
        raise ValueError("Unknown format: {}".format(format_file))



def read_toast_h_maps(
    list_name_files: list[str],
    list_spin: list[int] = [0, 1, 2, 3, 4],
    list_weights: np.ndarray = None,
    mask: np.ndarray | enmap.ndmap | None = None,
    format_file: str = "hdf",
    reweight_by_hits: bool = True,
    single_precision: bool = False,
    return_only_positive_spins: bool = True,
    return_attributes: bool = True,
) -> Spin_maps | tuple[Spin_maps, dict]:
    r"""
    Read the $h$-maps produced by sotodlib/TOAST, one file per detector, which 
    filename is preferrentially given as
        filever_w14_p082_LAT_f150_A.hdf
    with:
    * filever: the version of the file
    * w14: the number of the wafer (here 14)
    * p082: the pixel number (here 082)
    * LAT: the telescope used to generate the $h$-maps (here LAT)
    * f150: the frequency band (here 150 GHz)
    * A: the polarization detector number (here A)

    Parameters
    ----------
    list_name_files: list[str]
        Paths of the files, one per detector, with or without the extension ``format_file``.
    list_spin: list[int]
        Spins $n \ge 0$ to read; must contain 0, the hits weighting the detectors.
    list_weights: np.ndarray, optional
        Weights of the detectors, shape ``(n_det,)``. Default:the same weights 
        for all the detectors.
    mask: np.ndarray | enmap.ndmap, optional
        Mask of the full map (flattened or not; HEALPix in RING ordering): only the pixels
        where it is non-zero are read. Default: all the pixels.
    format_file: str
        Extension of the files: 'hdf', 'h5', 'hdf5' or 'H5'. Default is 'hdf'.
    reweight_by_hits: bool, optional
        If True (default), normalise the $h$-maps as smarties expects them:
        $h^d_n = w_d\,\mathrm{hits}_d \langle e^{in\psi}\rangle_d / \sum_{d'} w_{d'}\,\mathrm{hits}_{d'}$,
        so that $\sum_d h^d_0 = 1$. Otherwise $h^d_n = w_d \langle e^{in\psi}\rangle_d$ and
        $h^d_0 = w_d\,\mathrm{hits}_d$.
    single_precision: bool, optional
        If True, read the $h$-maps in single precision (complex64). Default is False.
    return_only_positive_spins: bool, optional
        If True (default), only store the spins of ``list_spin``; otherwise also store their
        opposites, $h_{-n} = \overline{h_n}$.
    return_attributes: bool, optional
        If True, also return the attributes of the files. Default is True.

    Returns
    -------
    h_maps: Spin_maps
        $h$-maps ``n -> (n_det, n_pix)``, with ``projection_pixel``, ``shape_fullsky`` and
        (CAR) ``wcs_car`` set; HEALPix maps in RING ordering.
    attributes: dict, optional
        If ``return_attributes``: for each attribute of the files, its values for the
        detectors (in the order of ``list_name_files``), as an array, e.g.
        ``attributes["pol_angle_rad"]``, ``attributes["wafer_x_mm"]``, ``attributes["det"]``.
    """
    assert format_file in FORMATS_H5, f"Format file must be one of {FORMATS_H5} for TOAST h-maps"
    assert np.unique(list_spin).size == len(list_spin), "The list of spins must be unique"
    assert 0 in list_spin and np.all(np.array(list_spin) >= 0), "The spins must be positive and contain 0, the hits weighting the detectors"

    dtype = np.complex64 if single_precision else np.complex128
    error_precision = 1e-8 if single_precision else 1e-15
    n_det = len(list_name_files)
    if list_weights is None:
        list_weights = np.ones(n_det) 
    else:
        list_weights = np.asarray(list_weights, dtype=float)
        assert list_weights.shape[0] == n_det, (
            "list_weights must be a 1D array of size n_det"
        )
        if np.sum(list_weights) != 1:
            list_weights = list_weights / np.sum(list_weights)


    h_n_spinmaps, attributes, geometry = None, {}, None
    flattened_indices_mask = ... if mask is None else np.flatnonzero(np.asarray(mask).ravel())

    for j, name in enumerate(list_name_files):
        with h5py.File(_file_name(name, format_file), "r") as file:
            if geometry is None:
                # This condition is entered with the first file,
                # so that the geometry and objects are defined only once
                geometry = _read_geometry(file)
                
                n_pix = int(np.prod(geometry["shape_fullsky"])) if mask is None else flattened_indices_mask.size
                h_n_spinmaps = Spin_maps.from_dictionary(
                    {spin: np.zeros((n_det, n_pix), dtype=dtype) for spin in list_spin}
                )
            # hits = _read_dataset(file, "0,0", "Re", geometry["to_ring"])[flattened_indices_mask]
            for spin in list_spin:
                value = _read_dataset(file, f"{spin},0", "Re", geometry["to_ring"])[flattened_indices_mask] + 1j * _read_dataset(file, f"{spin},0", "Im", geometry["to_ring"])[flattened_indices_mask]
                # value = np.where(hits > 0, value, 0)
                h_n_spinmaps[spin][j] = value * list_weights[j]
            for key, value in file.attrs.items():
                attributes.setdefault(key, [None] * n_det)[j] = value.decode() if isinstance(value, bytes) else value

    h_n_spinmaps.projection_pixel = geometry["projection_pixel"]
    h_n_spinmaps.shape_fullsky = geometry["shape_fullsky"]
    h_n_spinmaps.wcs_car = geometry["wcs"] if geometry["projection_pixel"] == "car" else None

    if not return_only_positive_spins:
        for spin in list_spin:
            if spin != 0:
                h_n_spinmaps[-spin] = np.conj(h_n_spinmaps[spin])

    if reweight_by_hits:
        reweight_hmaps_by_hits(
            h_dictionary=h_n_spinmaps,
            in_place=True,
        )

    if return_attributes:
        return h_n_spinmaps, {key: np.array(values) for key, values in attributes.items()}
    return h_n_spinmaps


def read_lbs_h_maps(
    list_detector_names: list[str],
    prefix: str = "h_maps_det_",
    list_spin: list[int] = [np.array([2, 0]), np.array([4, 0])],
    list_weights=None,
    mask: np.ndarray = None,
    format_file: str = "h5",
    reweight_by_hits: bool = True,
    return_only_positive_spins: bool = True,
    ignore_hwp_spins: bool = True,
    single_precision: bool = False,
    return_attributes: bool = True,
) -> Spin_maps:
    """
    Read the $h_{n,m}$ map as generated by litebird_sim,
    which filename is preferrentially given as
        h_maps_det_*.h5
    with:
        * the wafer number, detector id, frequency band, etc.

    Parameters
    ----------
    list_detector_names: list[str]
        The names of the detectors for which to read the $h_{n,m}$ maps.
    prefix: str
        Prefix of the $h_{n,m}$ maps files to read, typically 'h_maps_det_' for litebird_sim h maps
    list_spin: list[int]=[(2,0),(4,0)],
        List of spins to read
    list_weights: list[float], optional
        List of the weights per detector, to apply to the $h_{n,m}$ so that the total $h_{n,m}$ maps are weighted by the total nhits of the detectors. 
        If None, the nhits is assumed to be the same for all detectors and a weight of 1 is applied, except if reweight_by_hits is True, in which case the weights are computed from the number of pixels with non-zero $h_{n,m}$ values
        or by h_{0,0} values.
    mask: np.ndarray, optional
        HEALPix mask to define the area of the sky to compute the $h_{n,m}$ maps. If None, the full maps are read.
    format_file: str
        Format of the file to read, must be 'h5', 'hdf5', 'H5' or 'hdf', otherwise a ValueError is raised, as other formats are not supported for litebird_sim h maps
    reweight_by_hits: bool, optional
        If True, the weights are computed from the number of pixels with non-zero $h$-maps values, so that the total $h$-maps are weighted by the number of detectors observing each pixel.
    return_only_positive_spins: bool, optional
        If True, only load the positive spins from list_spin. Default is False.
    single_precision: bool, optional
        If True, read the $h_{n,m}$ maps in single precision (float32/complex64). If False, read in double precision (float64/complex128). Default is False.

    Returns
    -------
    h_maps: Spin_maps
        The $h$-maps stored in a dictionary of spins and ordered as [n_det, n_pix] per spin. 
    """
    try:
        from tqdm import tqdm
    except ImportError:
        raise ImportError(
            'tqdm is not installed. Please install it with "pip install tqdm"'
        )

    list_spin = [Spin_nm(spin) for spin in list_spin]

    assert np.array([spin[0] for spin in list_spin]).min() >= 0, (
        "The pointing spins provided must be positive, their negative counterpart will be computed from the components read"
    )

    assert (
        np.unique(np.atleast_2d(list_spin).sum(axis=1)).size
        == np.array(list_spin).shape[0]
    ), "The list of spins must be unique"
    assert (np.atleast_2d(list_spin)[0] >= 0).all(), (
        "The pointing spins provided must be positive, their negative counterpart will be computed from the components read"
    )
    assert np.array(list_spin).ndim == 1 or (
        np.array(list_spin).ndim == 2 and np.array(list_spin).shape[1] == 2
    ), (
        "The list of spins must be a 1D array of (pointing) spins or a 2D array of shape (n_spins, 2) for (n,m) spins"
    )
    assert format_file in ["h5", "hdf5", "H5", "hdf"], (
        "Format file must be 'h5', 'hdf5', 'H5' or 'hdf' for litebird_sim h maps"
    )

    projection_pixel = "healpix"  # Currently only healpix is supported for lbs h maps

    if single_precision:
        dtype = np.complex64
        error_precision = 1e-8
    else:
        dtype = np.complex128
        error_precision = 1e-15

    n_det = len(list_detector_names)

    if list_weights is None:
        list_weights = np.ones(n_det, dtype=float) 
    else:
        list_weights = np.asarray(list_weights, dtype=float)
        assert list_weights.shape[0] == n_det, (
            "list_weights must be a 1D array of size n_det"
        )
        if np.sum(list_weights) != 1:
            list_weights = list_weights / np.sum(list_weights)

    if mask is None:
        boolean_mask = ...
        fake_h_map = read_file(
            prefix + list_detector_names[0],
            format_file=format_file,
            projection_pixel=projection_pixel,
        )["{},{}".format(0, 0)]["Re"].ravel()
        nside = hp.npix2nside(fake_h_map.size)
        n_pix = np.prod(fake_h_map.shape)
    else:
        boolean_mask = np.bool_(mask)
        n_pix = boolean_mask[boolean_mask].size
        nside = hp.npix2nside(np.prod(mask.shape))

    total_list_spin = deepcopy(list_spin) 
    if not return_only_positive_spins:
        total_list_spin += [-spin for spin in list_spin if spin[0] > 0]

    spin_to_retain = 0 if ignore_hwp_spins else ...
    if ignore_hwp_spins:
        logger.info(
            "Ignoring HWP angle spins, removing m spins from the output...",
        )


    h_n_spinmaps = Spin_maps.from_dictionary(
        {spin[spin_to_retain]: np.zeros((n_det, n_pix), dtype=dtype) for spin in total_list_spin}
    )  # Initialize dictionary for spins
    shape_fullsky = (
        hp.nside2npix(nside),
    )  # Sufficient as only healpix maps are currently supported for lbs h maps

    flattened_indices_mask = ... if mask is None else np.flatnonzero(np.asarray(mask).ravel())
    attributes = {}
    geometry = None

    spin_0 = 0 if ignore_hwp_spins else Spin_nm((0, 0))

    for j, name_file in enumerate(tqdm(list_detector_names)):

        with h5py.File(_file_name(name_file, format_file), "r") as file:
            hits = _read_dataset(file, "0,0", "Re", to_ring=None)[flattened_indices_mask]
            for spin in list_spin:
                value = _read_dataset(file, f"{spin[0]},{spin[1]}", "Re", to_ring=None)[flattened_indices_mask] + 1j * _read_dataset(file, f"{spin[0]},{spin[1]}", "Im", to_ring=None)[flattened_indices_mask]
                h_n_spinmaps[spin[spin_to_retain]][j, ...].real = (
                    np.where(value.real != hp.UNSEEN, value.real, 0.0) * list_weights[j]
                )
                h_n_spinmaps[spin[spin_to_retain]][j, ...].imag = (
                    np.where(value.imag != hp.UNSEEN, value.imag, 0.0) * list_weights[j]
                )
            for key, value in file.attrs.items():
                attributes.setdefault(key, [None] * n_det)[j] = value.decode() if isinstance(value, bytes) else value

    h_n_spinmaps.projection_pixel = "healpix"
    h_n_spinmaps.shape_fullsky = hp.nside2npix(nside)
    h_n_spinmaps.wcs_car = None

    if not return_only_positive_spins:
        for spin in list_spin:
            h_n_spinmaps[-spin] = np.conj(h_n_spinmaps[spin])

    logger.info("Finalizing h_n dictionary...")

    print(spin_0, h_n_spinmaps.spins)
    if reweight_by_hits:
        reweight_hmaps_by_hits(
            h_dictionary=h_n_spinmaps,
            in_place=True,
            spin_0=spin_0
        )

    if return_attributes:
        return h_n_spinmaps, {key: np.array(values) for key, values in attributes.items()}
    return h_n_spinmaps


def build_mask_from_toast_h_maps(
    list_name_files: list[str],
    format_file: str = "hdf",
    return_detectors_hits: bool = False,
    return_original_shape: bool = False,
) -> np.ndarray | enmap.ndmap | tuple:
    """
    Mask of the pixels observed by at least one detector, from the hits of the $h$-maps
    produced by sotodlib/TOAST (group ``"0,0"``).

    Parameters
    ----------
    list_name_files: list[str]
        Paths of the files, one per detector, with or without the extension ``format_file``.
    format_file: str
        Extension of the files: 'hdf', 'h5', 'hdf5' or 'H5'. Default is 'hdf'.
    return_detectors_hits: bool, optional
        If True, also return the number of detectors observing each pixel. Default is False.
    return_original_shape: bool, optional
        If True, also return the shape of the full map, ``(ny, nx)`` for CAR, ``(npix,)``
        for HEALPix. Default is False.

    Returns
    -------
    mask: np.ndarray | enmap.ndmap
        Flattened mask (int8), 1 in the pixels observed by at least one detector; for CAR,
        an ``enmap.ndmap`` carrying the WCS; for HEALPix, in RING ordering.
    detectors_hits: np.ndarray | enmap.ndmap, optional
        Number of detectors observing each pixel, if ``return_detectors_hits``.
    shape_fullsky: tuple, optional
        Shape of the full map, if ``return_original_shape``.
    """
    assert format_file in FORMATS_H5, f"Format file must be one of {FORMATS_H5} for TOAST h-maps"

    detectors_hits, geometry = None, None
    for name in list_name_files:
        with h5py.File(_file_name(name, format_file), "r") as file:
            geometry = _read_geometry(file) if geometry is None else geometry
            observed = _read_dataset(file, "0,0", "Re", geometry) > 0
        detectors_hits = observed.astype(np.int64) if detectors_hits is None else detectors_hits + observed

    as_map = (lambda x: x) if geometry["wcs"] is None else (lambda x: enmap.enmap(x, wcs=geometry["wcs"]))
    result = (as_map(np.int8(detectors_hits > 0)),)
    if return_detectors_hits:
        result += (as_map(detectors_hits),)
    if return_original_shape:
        result += (geometry["shape_fullsky"],)
    return result[0] if len(result) == 1 else result


def build_mask_from_lbs_h_maps(
    list_name_files: list[str],
    format_file: str = "h5",
    spin_to_read: int | tuple = (0, 0),
    return_detectors_hits: bool = False,
    projection_pixel: str = "healpix",
    single_precision: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Read the $h_{n,m}$ map as generated by litebird_sim,
    which filename is preferrentially given as
        h_maps_det_*.h5
    with:
        * the wafer number, detector id, frequency band, etc.


    Parameters
    ----------
    list_name_files: list[str]
        The names of the detectors for which to read the $h_{n,m}$ maps.
    format_file: str
        Format of the file to read, must be 'h5', 'hdf5', 'H5' or 'hdf', otherwise a ValueError is raised, as other formats are not supported for litebird_sim h maps
    spin_to_read: int | tuple
        Spin to read, from which the mask will be built. If a tuple is provided, it is assumed to be a (n,m) spin for the $h_{n,m}$ maps
    return_detectors_hits: bool, optional
        If True, return the number of detectors observing each pixel in addition to the mask. Default is False.
    projection_pixel: str, optional
        Projection of the pixelization of the map to be read, either 'healpix' or 'car'. Default is 'healpix'.
    single_precision: bool, optional
        If True, read the $h_{n,m}$ maps in single precision (float32/complex64). If False, read in double precision (float64/complex128). Default is False.


    Returns
    -------
    h_n: Spin_maps
        The $h$-maps stored in a dictionary of spins and ordered as [n_det, n_pix], with n_det=1, except for spin=0 which dimension is [1,1] and only contains the float 1.

    Note:
    """
    try:
        from tqdm import tqdm
    except ImportError:
        raise ImportError(
            'tqdm is not installed. Please install it with "pip install tqdm"'
        )

    assert format_file in ["h5", "hdf5", "H5", "hdf"], (
        "Format file must be 'h5', 'hdf5', 'H5' or 'hdf' for litebird_sim h maps"
    )

    assert projection_pixel == "healpix", (
        "Currently only healpix projection is supported for litebird_sim h maps"
    )
    
    dtype = np.float32 if single_precision else np.float64

    mask = np.zeros_like(
        read_file(
            list_name_files[0],
            format_file=format_file,
            projection_pixel=projection_pixel,
        )["{},{}".format(*spin_to_read)]["Re"][:].ravel(),
        dtype=dtype,
    ).ravel()
    mask_detectors = np.int64(np.bool_(mask))

    for name_file in tqdm(list_name_files):
        h_map_file = read_file(
            name_file=name_file,
            format_file=format_file,
            projection_pixel=projection_pixel,
        )["{},{}".format(*spin_to_read)]["Re"][:].ravel()

        mask_contribution = np.where(h_map_file > 0, np.ones_like(h_map_file), 0)
        mask += mask_contribution
        mask_detectors += np.int64(np.bool_(mask_contribution))

    if return_detectors_hits:
        return np.int8(np.bool_(mask)), mask_detectors
    return np.int8(np.bool_(mask))


def read_ellipticity_values_from_yaml_file(
    path_values_ellipticity: str,
    transform_into_dict: bool = False,
    detector_features_all: list[str] = None,
) -> dict | dict[np.ndarray]:
    """
    Read the ellipticity values from a YAML file and return them as a dictionary of the input file
    or a dictionary with keys as the ellipticity parameters and values as arrays of the corresponding values for each detector.

    Parameters
    ----------
    path_values_ellipticity: str
        Path to the YAML file containing the ellipticity values.
    transform_into_dict: bool, optional
        If True, transform the ellipticity values into a dictionary with keys as the ellipticity parameters and values as arrays of the corresponding values for each detector. 
        Otherwise, return the ellipticity values as a dictionary read directly from the YAML file. Default is False.
    detector_features_all: list[str], optional
        List of detector names to extract the ellipticity values for. Required if transform_into_dict is True.
    
    Returns
    -------
    dict or dict of np.ndarray
        If transform_into_dict is False, returns the ellipticity values as a dictionary read directly from the YAML file.
        Otherwise, returns a dictionary with keys as the ellipticity parameters and values as arrays of the corresponding values for each detector.
    """
    if transform_into_dict:
        assert detector_features_all is not None, (
            "detector_features_all must be provided if transform_into_array is True"
        )

    with open(path_values_ellipticity) as file:
        ellipticity_file = yaml.safe_load(file)

    if transform_into_dict:
        ellipticity_parameters_dict = {
            key: np.array(
                [ellipticity_file[det_name][key] for det_name in detector_features_all]
            )
            for key in ellipticity_file[list(ellipticity_file.keys())[-1]].keys()
        }
        ellipticity_parameters_dict["ellipticity_parameter_convention"] = (
            ellipticity_file["ellipticity_parameter_convention"]
        )
        return ellipticity_parameters_dict
    return ellipticity_file


def read_values_from_yaml_file(
    path_yaml: str,
    transform_into_dict: bool = False,
    detector_features_all: list[str] = None,
    retrieve_all_parameters: bool = False,
    list_additional_parameters_to_retrieve: list[str] = None,
) -> dict | dict[np.ndarray]:
    """
    Read the values from a YAML file and return them as a dictionary of the input file
    or a dictionary with keys as the parameters and values as arrays of the corresponding values for each detector.
    """
    if transform_into_dict:
        assert detector_features_all is not None, (
            "detector_features_all must be provided if transform_into_array is True"
        )

    with open(path_yaml) as file:
        dictionary_yaml = yaml.safe_load(file)

    if not transform_into_dict:
        return dictionary_yaml

    else:
        list_keys_parameter_values = dictionary_yaml[detector_features_all[0]].keys()
        output_parameters_dict = {
            key: np.array(
                [dictionary_yaml[det_name][key] for det_name in detector_features_all]
            )
            for key in list_keys_parameter_values
        }

        if retrieve_all_parameters:
            other_parameters = list(dictionary_yaml.keys())
            other_parameters.remove(detector_features_all)

            for key in other_parameters:
                output_parameters_dict[key] = dictionary_yaml[key]
        if list_additional_parameters_to_retrieve is not None:
            for key in list_additional_parameters_to_retrieve:
                if key in dictionary_yaml:
                    output_parameters_dict[key] = dictionary_yaml[key]
        return output_parameters_dict


def read_spherical_derivatives_from_file(
    path_spherical_derivatives: str,
    is_car: bool = False,
) -> dict[str, np.ndarray]:
    """
    Read the spherical derivatives from a file and return them as a dictionary of the input file
    or a dictionary with keys as the derivatives and values as arrays of the corresponding values for each derivative.
    The ordering is assumed to be [no_deriv, theta, phi, theta_theta, theta_phi, phi_phi] for the derivatives,
    with no_deriv being the first map not derived for and the rest being the derivatives. 
    """
    wcs_car = None

    if os.path.isfile(path_spherical_derivatives):
        if not is_car:
            spherical_derivatives_array = hp.read_map(
                path_spherical_derivatives, field=None
            )
        else:
            spherical_derivatives_array = flatten_CAR_maps(
                enmap.read_map(path_spherical_derivatives)
            )
            wcs_car = spherical_derivatives_array.wcs
        assert spherical_derivatives_array.shape[0] == (6), (
            "The spherical derivatives file must contain 6 maps with shape (6, npix)."
        )

        list_derivatives = ["theta", "phi", "theta_theta", "theta_phi", "phi_phi"]

        get_final_map = lambda x: x if wcs_car is None else enmap.ndmap(x, wcs=wcs_car)

        return {
            derivative: get_final_map(spherical_derivatives_array[i + 1])
            for i, derivative in enumerate(list_derivatives)
        }


def save_partial_spin_maps(
    partial_spin_maps: list[np.ndarray],
    nstokes: int,
    shape_pixels: tuple[int, ...],
    mask_on_full_map: np.ndarray,
    path_output: str,
    format_output: str = ".npy",
    dtype: type = complex
) -> None:
    """
    Save the partial spin maps to a file in the specified format.
    """

    extended_final_maps = np.zeros((nstokes,) + (np.prod(shape_pixels),), dtype=dtype)
    if nstokes == 3 or nstokes == 1:
        extended_final_maps[0, mask_on_full_map != 0] = partial_spin_maps[0]
    if nstokes == 3 or nstokes == 2:
        final_Q_map = (partial_spin_maps[-2] + partial_spin_maps[2]) / 2.0
        final_U_map = 1j * (partial_spin_maps[-2] - partial_spin_maps[2]) / 2.0

        extended_final_maps[-2, mask_on_full_map != 0] = final_Q_map.real
        extended_final_maps[-1, mask_on_full_map != 0] = final_U_map.real

    if extended_final_maps.shape[-len(shape_pixels) :] != shape_pixels:
        extended_final_maps = extended_final_maps.reshape(
            extended_final_maps.shape[:-1] + shape_pixels
        )

    is_car = False
    first_spin = 0 if (nstokes == 1 or nstokes == 3) else -2
    if type(partial_spin_maps[first_spin]) == enmap.ndmap:
        extended_final_maps = enmap.ndmap(
            extended_final_maps.astype(np.float32 if dtype == np.complex64 else np.float64), 
            wcs=partial_spin_maps[first_spin].wcs,
        )
        is_car = True

    if is_car:
        if format_output == ".fits" and not path_output.endswith(".fits"):
            path_output = path_output + ".fits"
        elif format_output == ".hdf" and not path_output.endswith(".hdf"):
            path_output = path_output + ".hdf"
        enmap.write_map(path_output, extended_final_maps, extra={"BUNIT": "uK"})
    elif format_output == ".npy" or path_output.endswith(".npy"):
        if not path_output.endswith(".npy"):
            path_output = path_output + ".npy"
        logger.info(f"Saving map into {path_output}")
        np.save(path_output, extended_final_maps[:, mask_on_full_map != 0])
    elif format_output == ".fits" or path_output.endswith(".fits"):
        if not path_output.endswith(".fits"):
            path_output = path_output + ".fits"
        logger.info(f"Saving map into {path_output}")

        hp.write_map(path_output, extended_final_maps, overwrite=True)

    elif (
        format_output in [".hdf", ".hdf5"]
        or path_output.endswith(".hdf")
        or path_output.endswith(".hdf5")
    ):
        if not (path_output.endswith(".hdf") or path_output.endswith(".hdf5")):
            path_output = path_output + ".hdf"
        logger.info(f"Saving map into {path_output}")

        with h5py.File(path_output, "w") as hf:
            hf.create_dataset("maps", data=extended_final_maps)
        hf.close()
    else:
        raise ValueError(
            "Unsupported format_output. Supported formats are '.npy' and '.fits'."
        )



def read_toast_h_maps_old(
    list_name_files: list[str],
    list_spin: list[int] = [0, 2, 4],
    list_prefix: list[str] = ["A_", "B_"],
    list_weights=None,
    mask: np.ndarray | enmap.ndmap | None = None,
    format_file: str = "npz",
    reweight_by_hits: bool = True,
    single_precision: bool = False,
    duplicate_maps_with_prefix=False,
    return_only_positive_spins: bool = True,
    projection_pixel: str = "healpix",
) -> Spin_maps:
    """
    TO BE DEPRECATED: Only kept for backward compatibility with old TOAST h-maps files, which are not compatible with the new read_h_maps function.

    Read the $h$-maps as generated by TOAST,
    which filename is preferrentially given as
        filever_w14_p082_f150_A_sin_1.fits
    with:
    * filever: the version of the file
    * w14: the number of the wafer (here 14)
    * p082: the pixel number (here 082)
    * f150: the frequency band (here 150 GHz)
    * A: the polarization detector number (here A)
    * sin_1: the spin dependence and component written, here the $h$-maps component with spin 1

    Parameters
    ----------
    list_name_files: list[str]
        List of the names of the files to read, without the suffix corresponding to the format indicated in format_file.
    list_spin (list[int]):
        List of the spin to read
    list_prefix: list[str]
        List of the prefixes to read, typically ['A_', 'B_'] for two polarization detectors in a pair
    list_weights: list[float], optional
        List of the weights per detector, to apply to the $h$-maps so that the total $h$-maps are weighted by the total nhits of the detectors.
        If None, the nhits is assumed to be the same for all detectors and a weight of 1 is applied, except if reweight_by_hits is True, in which case the weights are computed 
        from the read h_0 maps or the number of pixels with non-zero $h$-maps values.
    mask: np.ndarray | enmap.ndmap, optional
        Mask to define the area of the sky to compute the $h$-maps. If None, the full maps are read.
    format_file: str
        Format of the file to read, must be in ["npy", "npz", "h5", "hdf5", "H5", "hdf", "fits"], otherwise a ValueError is raised
    reweight_by_hits: bool, optional
        If True, the weights are computed from the number of hits per pixel and per detectors if available, are read from spin 0 maps, otherwise by the number of detectors observing each pixel.
    single_precision: bool, optional
        If True, read the $h$-maps in single precision (float32). If False, read in double precision (float64). Default is False.
    duplicate_maps_with_prefix: bool, optional
        If True, duplicate the maps for each prefix in list_prefix (so that the h-maps for each pair are identical). Default is False.
    return_only_positive_spins: bool, optional
        If True, only load the positive spins from list_spin. Default is False.
    projection_pixel: str, optional
        Projection of the pixelization of the map to be read, either 'healpix' or 'car'. Default is 'healpix'.

    Returns
    -------
    h_maps: Spin_maps
        The $h$-maps stored in a dictionary of spins and ordered as [n_det, n_pix].
    """
    try:
        from tqdm import tqdm
    except ImportError:
        raise ImportError(
            'tqdm is not installed. Please install it with "pip install tqdm"'
        )

    assert np.unique(list_spin).size == np.array(list_spin).size, (
        "The list of spins must be unique"
    )
    assert (np.array(list_spin) >= 0).all(), (
        "The spins provided must be positive, their negative counterpart will be computed from the components read"
    )

    assert format_file in ["npy", "npz", "h5", "hdf5", "H5", "hdf", "fits"], (
        "Format file must be 'fits', 'npy', 'h5', 'hdf5', 'H5', 'hdf' or 'npz' for TOAST h maps"
    )

    if single_precision:
        dtype = np.complex64
        error_precision = 1e-8
    else:
        dtype = np.complex128
        error_precision = 1e-15

    if duplicate_maps_with_prefix:
        logger.info("Duplicate maps with prefix is enabled")

    number_detectors_prefix = len(list_prefix)
    n_det = len(list_name_files) * number_detectors_prefix

    if list_weights is None:
        list_weights = np.ones(n_det, dtype=float) / n_det
    else:
        list_weights = np.asarray(list_weights, dtype=float)
        # assert list_weights.shape[0] == n_det and list_weights.ndim <= 2, (
        #     "list_weights must be a 1D array of size n_det or a 2D array of shape (n_det, n_pix)"
        # )
        assert list_weights.shape[0] == n_det, (
            "list_weights must be a 1D array of size n_det"
        )
        if np.sum(list_weights) != 1:
            list_weights = list_weights / np.sum(list_weights)


    if mask is None:
        boolean_mask = ...
        fake_h_map = read_file(
            list_name_files[0]
            + "_{}_sin_{}".format(list_prefix[0].replace("_", ""), list_spin[0]),
            format_file=format_file,
            boolean_mask=boolean_mask,
            projection_pixel=projection_pixel,
        )
        n_pix = np.prod(fake_h_map[boolean_mask].shape)
    else:
        boolean_mask = np.bool_(mask)
        n_pix = boolean_mask[boolean_mask].size

    # Setting geometry
    fake_h_map = read_file(
        list_name_files[0]
        + "_{}_sin_{}".format(list_prefix[0].replace("_", ""), list_spin[0]),
        format_file=format_file,
        boolean_mask=None,
        projection_pixel=projection_pixel,
        flatten_output=False,
    )
    wcs = None if projection_pixel == "healpix" else fake_h_map.wcs
    shape_fullsky = fake_h_map.shape
    if shape_fullsky[0] == 1:
        shape_fullsky = shape_fullsky[1:]
    fake_h_map = None

    total_list_spin = deepcopy(list_spin)
    if not return_only_positive_spins:
        total_list_spin += [-spin for spin in list_spin]

        if 0 in total_list_spin and total_list_spin.count(0) > 1:
            total_list_spin.remove(0)

    h_n_spinmaps = Spin_maps.from_dictionary(
        {spin: np.zeros((n_det, n_pix), dtype=dtype) for spin in total_list_spin}
    )  # Initialize dictionary for spins
    h_n_spinmaps.projection_pixel = projection_pixel
    h_n_spinmaps.wcs_car = wcs
    h_n_spinmaps.shape_fullsky = shape_fullsky

    for j, name_file in enumerate(tqdm(list_name_files)):
        spin_h_n = dict()
        if not duplicate_maps_with_prefix:
            for prefix in list_prefix:
                for spin in list_spin:
                    for component in ["sin", "cos"]:
                        spin_h_n[prefix + f"{component}_{spin}"] = read_file(
                            name_file
                            + f"_{prefix.replace('_', '')}_{component}_{spin}",
                            format_file=format_file,
                            boolean_mask=boolean_mask,
                            projection_pixel=projection_pixel,
                        )
        else:
            for spin in list_spin:
                for component in ["sin", "cos"]:
                    spin_h_n[list_prefix[0] + f"{component}_{spin}"] = read_file(
                        name_file
                        + f"_{list_prefix[0].replace('_', '')}_{component}_{spin}",
                        format_file=format_file,
                        boolean_mask=boolean_mask,
                        projection_pixel=projection_pixel,
                    )
            for prefix in list_prefix[1:]:
                for spin in list_spin:
                    for component in ["sin", "cos"]:
                        spin_h_n[prefix + f"{component}_{spin}"] = spin_h_n[
                            list_prefix[0] + f"{component}_{spin}"
                        ]

        for spin in list_spin:
            for count_prefix, prefix in enumerate(list_prefix):
                sin_spin_h_n = spin_h_n[prefix + "sin_{}".format(spin)]
                cos_spin_h_n = spin_h_n[prefix + "cos_{}".format(spin)]

                h_n_spinmaps[spin][
                    j * number_detectors_prefix + count_prefix, ...
                ].real = cos_spin_h_n * list_weights[j]
                h_n_spinmaps[spin][
                    j * number_detectors_prefix + count_prefix, ...
                ].imag = sin_spin_h_n * list_weights[j]
        spin_h_n = None

    if not return_only_positive_spins:
        for spin in list_spin:
            h_n_spinmaps[-spin] = np.conj(h_n_spinmaps[spin])

    logger.info("Finalizing h_n dictionary...")

    if reweight_by_hits:
        reweight_hmaps_by_hits(
            h_dictionary=h_n_spinmaps,
            in_place=True
        )

    return h_n_spinmaps


def build_mask_from_toast_h_maps_old(
    list_name_files: list[str],
    list_prefix: list[str] = ["A_", "B_"],
    format_file: str = "npz",
    spin_to_read: int = 1,
    return_detectors_hits: bool = False,
    projection_pixel: str = "healpix",
    return_original_shape: bool = False,
    single_precision: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, tuple]:
    """
    TO BE DEPRECATED: Only kept here for backward compatibility, use build_mask_from_toast_h_maps instead.

    Read the $h$-maps as generated by TOAST,
    which filename is preferrentially given as
        filever_w14_p082_f150_A_sin_1.fits
    with:
    * filever: the version of the file
    * w14: the number of the wafer (here 14)
    * p082: the pixel number (here 082)
    * f150: the frequency band (here 150 GHz)
    * A: the polarization detector number (here A)
    * sin_1: the spin dependence and component written, here the $h$-maps component with spin 1

    Only the latest part of the file actually matters, considering the $sin$ or $cos$ component of the $h$-maps as well as its spin dependence.

    Parameters
    ----------
    list_name_files: list[str]
        List of the names of the files to read per detector, without the suffix corresponding to the format indicated in format_file.
    list_prefix: list[str]
        List of the prefixes to read, typically ['A_', 'B_'] for two polarization detectors in a pair
    format_file: str
        Format of the file to read, must be in ["npy", "npz", "h5", "hdf5", "H5", "hdf", "fits"], otherwise a ValueError is raised
    spin_to_read: int
        Spin to read, from which the mask will be built. 
    return_detectors_hits: bool, optional
        If True, return the number of detectors observing each pixel in addition to the mask. Default is False.
    projection_pixel: str, optional
        Projection of the pixelization of the map to be read, either 'healpix' or 'car'. Default is 'healpix'.
    return_original_shape: bool, optional
        If True, return the original shape of the maps read in addition to the mask. Default is False.
    single_precision: bool, optional
        If True, return the mask in single precision (float32). If False, return in double precision (float64). Default is False.


    Returns
    -------
    mask: np.ndarray | enmap.ndmap
        The mask built from the $h$-maps read, with 1 for pixels observed by at least one detector and 0 for pixels not observed by any detector.
    mask_detectors: np.ndarray | enmap.ndmap, optional
        The number of detectors observing each pixel, if return_detectors_hits is True.
    shape_fullsky: tuple, optional
        The original shape of the maps read, if return_original_shape is True.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        raise ImportError(
            'tqdm is not installed. Please install it with "pip install tqdm"'
        )

    assert (
        format_file in ["npy", "npz"]
        or format_file in ["h5", "hdf5", "H5", "hdf"]
        or format_file == "fits"
    )

    dtype = np.float32 if single_precision else np.float64

    fake_h_map = read_file(
        list_name_files[0]
        + "_{}_sin_{}".format(list_prefix[0].replace("_", ""), spin_to_read),
        format_file=format_file,
        boolean_mask=None,
        projection_pixel=projection_pixel,
        flatten_output=False,
    )
    wcs = None if projection_pixel == "healpix" else fake_h_map.wcs
    shape_fullsky = fake_h_map.shape
    if shape_fullsky[0] == 1:
        shape_fullsky = shape_fullsky[1:]

    mask = np.zeros_like(
        fake_h_map, 
        dtype=dtype).ravel()
    fake_h_map = None
    mask_detectors = np.int64(np.bool_(mask))

    for name_file in tqdm(list_name_files):
        for prefix in list_prefix:
            map_cos_sin = (
                np.abs(
                    read_file(
                        name_file
                        + "_{}_sin_{}".format(prefix.replace("_", ""), spin_to_read),
                        format_file=format_file,
                        projection_pixel=projection_pixel,
                    )[:]
                ).ravel()
                + np.abs(
                    read_file(
                        name_file
                        + "_{}_cos_{}".format(prefix.replace("_", ""), spin_to_read),
                        format_file=format_file,
                        projection_pixel=projection_pixel,
                    )[:]
                ).ravel()
            )
            mask += map_cos_sin
            mask_detectors += np.int64(np.bool_(map_cos_sin))

    if format_file in ["h5", "hdf5", "H5", "hdf"] and projection_pixel == "healpix":
        reorder_output = lambda x: hp.reorder(x, n2r=True)
    elif projection_pixel == "car" and wcs is not None:
        reorder_output = lambda x: enmap.enmap(x, wcs=wcs)
    else:
        reorder_output = lambda x: x

    result = (reorder_output(np.int8(np.bool_(mask))),)
    if not return_detectors_hits and not return_original_shape:
        return result[0]
    if return_detectors_hits:
        result = result + (reorder_output(mask_detectors),)
    if return_original_shape:
        result = result + (shape_fullsky,)
    return result

