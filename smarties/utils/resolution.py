# This file is part of SMARTIES.
# Copyright (C) 2024-2026 Members of the Simons Observatory collaboration.
# Please refer to the LICENSE file in the root of this repository.

from copy import deepcopy

import healpy as hp
import numpy as np
from pixell import enmap, wcsutils

from smarties.hn import Spin_maps

__all__ = [
    'get_chunks_from_shape',
    'get_projector_map_resolution',
    'get_chunks_final_map_from_downgraded_chunks',
    'get_slice_1d_from_2d',
    # Adaptive resolution, CAR and HEALPix
    'get_ratio_resolution',
    'get_parent_pixels_car',
    'get_parent_pixels_healpix',
    'get_parent_pixels',
    'get_children_pixels_healpix',
    'downgrade_mask',
    'get_chunks_healpix',
    'get_pixels_chunk',
    'get_projector_chunk',
    'upgrade_h_maps',
]

def get_chunks_from_shape(
    shape_map: tuple,
    number_chunks: int = 10,
) -> list:
    """
    Split a map into chunks based on its shape.

    Parameters
    ----------
    shape_map : tuple
        Shape of the map to be split.
    number_chunks : int, optional
        Number of chunks to split the map into. Default is 10.
        For a 2D map, either a pair ``(n_x, n_y)`` of numbers of chunks along each axis,
        or a scalar ``n``, giving ``(int(sqrt(n)) + 1)`` chunks along each axis.

    Returns
    -------
    np.ndarray
        Chunks, as ``(start, stop)`` for a 1D map and ``(start_x, stop_x, start_y, stop_y)``
        for a 2D map (``x`` being the first axis). The chunks cover the whole map.
    """

    def boundaries(size, number):
        number = int(max(1, min(int(number), int(size))))
        return np.linspace(0, int(size), number + 1).astype(np.int64)

    array_shape_map = np.array(shape_map)
    if len(array_shape_map) == 1:
        assert np.array(number_chunks).size == 1, (
            "Number of chunks should be a scalar for 1D maps."
        )
        limits = boundaries(array_shape_map[0], np.array(number_chunks).item())
        return np.int_([(limits[i], limits[i + 1]) for i in range(limits.size - 1)])

    elif len(array_shape_map) == 2:
        assert np.array(number_chunks).size < 3, (
            "Number of chunks should be a scalar or 2D array for 2D maps."
        )
        if np.array(number_chunks).size == 1:
            sqrt_number_chunks = int(np.sqrt(np.array(number_chunks).item())) + 1
            number_chunks_x, number_chunks_y = (sqrt_number_chunks, sqrt_number_chunks)
        else:
            number_chunks_x, number_chunks_y = number_chunks

        x_chunks = boundaries(array_shape_map[0], number_chunks_x)
        y_chunks = boundaries(array_shape_map[1], number_chunks_y)

        return np.int_(
            [
                (x_chunks[i], x_chunks[i + 1], y_chunks[j], y_chunks[j + 1])
                for i in range(len(x_chunks) - 1)
                for j in range(len(y_chunks) - 1)
            ]
        )


def get_projector_map_resolution(
    shape_map_downgraded: tuple,
    projection_pixel: str = "car",
    factor: int = 2,
    boolean_mask: bool = None,
    nest: bool = False,
    wcs_downgraded: wcsutils.WCS = None,
) -> enmap.ndmap | np.ndarray:
    """
    Get the projector map resolution for a given downgraded map, allowing to
    upgrade a downgraded map to a higher resolution map only through indexation.

    Parameters
    ----------
    shape_map_downgraded : tuple
        Shape of the downgraded map: ``(ny, nx)`` for CAR, ``(npix,)`` for HEALPix.
    projection_pixel : str, optional
        Type of pixelization used for the map. Options are 'healpix' or 'car'. Default is 'car'.
    factor : int, optional
        Factor by which to upgrade the resolution. Default is 2.
        CAR: an integer or a pair of integers (one per axis); HEALPix: the ratio of the
        nside, a power of 2.
    boolean_mask : bool, optional
        Boolean mask of the downgraded map, selecting the pixels held by the compressed
        downgraded maps. If None, all the pixels are selected. Default is None.
    nest : bool, optional
        HEALPix ordering of both maps: NESTED if True, RING otherwise. Default is False.
    wcs_downgraded : wcsutils.WCS
        WCS object of the downgraded map (CAR only, ignored for HEALPix).

    Returns
    -------
    enmap.ndmap or np.ndarray
        For each pixel of the upgraded (full) map, the index, in the compressed downgraded
        map, of the downgraded pixel containing it. The entries of the pixels whose
        downgraded pixel is not selected by ``boolean_mask`` are 0: restrict the projector
        to the upgraded pixels of selected downgraded pixels.
    """
    assert projection_pixel in ["healpix", "car"], (
        f"Unknown projection pixel: {projection_pixel}"+
        "should be either 'healpix' or 'car'."
    )
    npix_downgraded = int(np.prod(shape_map_downgraded))
    boolean_mask = (
        np.ones(npix_downgraded, dtype=bool) if boolean_mask is None
        else np.asarray(boolean_mask).ravel() != 0
    )
    npix = int(boolean_mask.sum())

    if projection_pixel == "healpix":
        assert np.array(factor).size == 1, (
            "For HEALPix, the factor should be a scalar, the ratio of the nside of the full map "
            "to the nside of the h-maps, a power of 2."
        )
        assert 2**int(np.log2(factor)) == factor, (
            "For HEALPix, the factor should be a power of 2."
        )
        ratio = int(np.prod(factor))
        nside_downgraded = hp.npix2nside(npix_downgraded)
        nside = nside_downgraded * ratio
        indices_map = np.zeros(npix_downgraded, dtype=np.int64)
        indices_map[boolean_mask] = np.arange(npix)

        parents = get_parent_pixels_healpix(
            np.arange(hp.nside2npix(nside)), nside, nside_downgraded, nest=nest
        )
        return indices_map[parents]
    else:
        indices_map = enmap.zeros(
            (shape_map_downgraded[0], shape_map_downgraded[1]),
            wcs=wcs_downgraded,
            dtype=np.int32,
        )
        indices_map.ravel()[boolean_mask] = np.arange(npix)

        return indices_map.upgrade(factor=factor).ravel()


def get_chunks_final_map_from_downgraded_chunks(
    shape_final_map: tuple,
    shape_map_downgraded: tuple,
    list_chunks_previous_map: list,
) -> list:
    """
    Get the chunks of the final map based on the chunks of a downgraded map.

    Parameters
    ----------
    shape_final_map : tuple
        Shape of the final map.
    shape_map_downgraded : tuple
        Shape of the downgraded map.
    list_chunks_previous_map : list
        List of chunks of the downgraded map.

    Returns
    -------
    list
        List of chunks of the final map.

    Notes
    -----
    For 1D (HEALPix) maps, a range of pixels of the downgraded map corresponds to a range
    of pixels of the final map in NESTED ordering only.
    """
    assert np.array(shape_final_map).size == np.array(shape_map_downgraded).size, (
        "Shape of the full map and the previous map should have the same number of dimensions."
    )

    if np.array(shape_final_map).size == 1:
        ratio = np.array(shape_final_map)[0] // np.array(shape_map_downgraded)[0]
        return np.array(list_chunks_previous_map) * ratio

    elif np.array(shape_final_map).size == 2:
        ratio_x = np.array(shape_final_map)[0] // np.array(shape_map_downgraded)[0]
        ratio_y = np.array(shape_final_map)[1] // np.array(shape_map_downgraded)[1]
        return np.array(list_chunks_previous_map) * np.array(
            [ratio_x, ratio_x, ratio_y, ratio_y]
        )


def get_slice_1d_from_2d(
    shape_map: tuple,
    slice_x: slice,
    slice_y: slice,
    boolean_mask: np.ndarray = None
) -> enmap.ndmap:
    """
    Get the 1D slice of a 2D CAR map flattened based on the provided slices in x and y directions.

    Parameters
    ----------
    shape_map : tuple
        Shape of the 2D map.
    slice_x : slice
        Slice object for the x direction.
    slice_y : slice
        Slice object for the y direction.
    boolean_mask : np.ndarray, optional
        Boolean mask to apply to the map. If None, no mask is applied. Default is None.

    Returns
    -------
    enmap.ndmap
        1D array of indices corresponding to the sliced 2D map.

    Notes
    -----
    This function builds a full-size index map at each call;
    :func:`get_projector_chunk` obtains the same indices without it.
    """
    if slice_y is ... or slice_x is ...:
        return slice_x if slice_x is not ... else slice_y
    indices_map = enmap.zeros(shape_map)
    indices_map[:] = np.arange(shape_map[0] * shape_map[1]).reshape(shape_map)

    if boolean_mask is not None:
        npix = np.prod(boolean_mask[boolean_mask].shape)
        indices_map[:] = -1
        indices_map.ravel()[boolean_mask.ravel()] = np.arange(npix)
    indices_output_1d = indices_map[slice_x, slice_y].ravel()
    return np.int_(indices_output_1d[indices_output_1d >= 0])



def get_ratio_resolution(
    projection_pixel: str,
    shape_final_map: tuple,
    shape_h_maps: tuple,
) -> tuple[int, ...]:
    """
    Ratio of resolution between the output map and the $h$-maps.

    Parameters
    ----------
    projection_pixel : str
        'car' or 'healpix'.
    shape_final_map : tuple
        Shape of the full output map: ``(ny, nx)`` for CAR, ``(npix,)`` for HEALPix.
    shape_h_maps : tuple
        Shape of the full $h$-map grid, in the same format.

    Returns
    -------
    tuple[int, ...]
        CAR: ``(ny // my, nx // mx)``; HEALPix: ``(nside // nside_h_maps,)``.
    """
    assert projection_pixel in ["healpix", "car"], (
        f"Unknown projection pixel: {projection_pixel}"
    )
    if projection_pixel == "car":
        assert len(shape_final_map) == 2 and len(shape_h_maps) == 2, (
            "CAR shapes must be 2D"
        )
        ratio_x, remainder_x = divmod(int(shape_final_map[0]), int(shape_h_maps[0]))
        ratio_y, remainder_y = divmod(int(shape_final_map[1]), int(shape_h_maps[1]))
        assert remainder_x == 0 and remainder_y == 0 and ratio_x >= 1 and ratio_y >= 1, (
            f"The output CAR shape {tuple(shape_final_map)} must be an integer multiple "
            f"of the h-map shape {tuple(shape_h_maps)}"
        )
        return ratio_x, ratio_y
    else:
        nside = hp.npix2nside(int(np.prod(shape_final_map)))
        nside_h_maps = hp.npix2nside(int(np.prod(shape_h_maps)))
        ratio, remainder = divmod(nside, nside_h_maps)
        assert remainder == 0 and ratio >= 1 and np.log2(ratio) - int(np.log2(ratio)) == 0, (
            f"The output nside ({nside}) must be the h-map nside ({nside_h_maps}) times a power of 2"
        )
        return (ratio,)


def get_parent_pixels_car(
    pixels: np.ndarray,
    shape_final_map: tuple,
    shape_h_maps: tuple,
) -> np.ndarray:
    """
    Index, in the flattened CAR grid of the $h$-maps, of the pixel containing each
    output pixel.

    Parameters
    ----------
    pixels : np.ndarray
        Indices of output pixels in the flattened output grid.
    shape_final_map : tuple
        Shape ``(ny, nx)`` of the output CAR grid.
    shape_h_maps : tuple
        Shape ``(my, mx)`` of the $h$-map CAR grid, ``ny`` and ``nx`` being integer
        multiples of ``my`` and ``mx``.

    Returns
    -------
    np.ndarray
        Indices in the flattened $h$-map grid, same shape as ``pixels``.
    """
    ratio_x, ratio_y = get_ratio_resolution("car", shape_final_map, shape_h_maps)
    row, column = np.divmod(np.asarray(pixels, dtype=np.int64), int(shape_final_map[1]))
    return (row // ratio_x) * int(shape_h_maps[1]) + column // ratio_y


def get_parent_pixels_healpix(
    pixels: np.ndarray,
    nside_final_map: int,
    nside_h_maps: int,
    nest: bool = False,
) -> np.ndarray:
    """
    Index of the HEALPix pixel at ``nside_h_maps`` containing each output pixel at
    ``nside_final_map``.

    In NESTED ordering, the parent of pixel $p$ is $p // r^2$ for
    ``nside_final_map = r * nside_h_maps``. Same as
    ``hp.ud_grade(np.arange(npix_h_maps), nside_final_map)[pixels]``, without building the
    full-sky map.

    Parameters
    ----------
    pixels : np.ndarray
        Output pixel indices.
    nside_final_map : int
        Resolution of the output map, ``nside_h_maps`` times a power of 2.
    nside_h_maps : int
        Resolution of the $h$-maps.
    nest : bool, optional
        Ordering of both maps: NESTED if True, RING otherwise (default).

    Returns
    -------
    np.ndarray
        Pixel indices at ``nside_h_maps``, in the same ordering.
    """
    (ratio,) = get_ratio_resolution(
        "healpix", (hp.nside2npix(int(nside_final_map)),), (hp.nside2npix(int(nside_h_maps)),)
    )
    pixels = np.asarray(pixels, dtype=np.int64)
    nested = pixels if nest else hp.ring2nest(int(nside_final_map), pixels)
    parents = nested // ratio**2
    return parents if nest else hp.nest2ring(int(nside_h_maps), parents)


def get_parent_pixels(
    projection_pixel: str,
    pixels: np.ndarray,
    shape_final_map: tuple,
    shape_h_maps: tuple,
    nest: bool = False,
) -> np.ndarray:
    """
    Index, in the full $h$-map grid, of the pixel containing each output pixel.

    Parameters
    ----------
    projection_pixel : str
        'car' or 'healpix'.
    pixels : np.ndarray
        Indices of output pixels in the full (flattened) output map.
    shape_final_map : tuple
        Shape of the full output map: ``(ny, nx)`` for CAR, ``(npix,)`` for HEALPix.
    shape_h_maps : tuple
        Shape of the full $h$-map grid, in the same format.
    nest : bool, optional
        HEALPix ordering: NESTED if True, RING otherwise (default).

    Returns
    -------
    np.ndarray
        Indices in the full (flattened) $h$-map grid.
    """
    if projection_pixel == "car":
        return get_parent_pixels_car(pixels, shape_final_map, shape_h_maps)
    return get_parent_pixels_healpix(
        pixels,
        hp.npix2nside(int(np.prod(shape_final_map))),
        hp.npix2nside(int(np.prod(shape_h_maps))),
        nest=nest,
    )


def get_children_pixels_healpix(
    pixels_h_maps: np.ndarray,
    nside_final_map: int,
    nside_h_maps: int,
    nest: bool = False,
) -> np.ndarray:
    """
    Pixels at ``nside_final_map`` contained in HEALPix pixels at ``nside_h_maps``.

    In NESTED ordering, the children of pixel $P$ are $P r^2, \\dots, P r^2 + r^2 - 1$ for
    ``nside_final_map = r * nside_h_maps``.

    Parameters
    ----------
    pixels_h_maps : np.ndarray
        Pixel indices at ``nside_h_maps``, shape ``(n,)``.
    nside_final_map : int
        Resolution of the output map, ``nside_h_maps`` times a power of 2.
    nside_h_maps : int
        Resolution of the $h$-maps.
    nest : bool, optional
        Ordering of both maps: NESTED if True, RING otherwise (default).

    Returns
    -------
    np.ndarray
        Children pixels, shape ``(n, (nside_final_map // nside_h_maps)**2)``, in the same
        ordering; row ``i`` holds the children of ``pixels_h_maps[i]``.
    """
    (ratio,) = get_ratio_resolution(
        "healpix", (hp.nside2npix(int(nside_final_map)),), (hp.nside2npix(int(nside_h_maps)),)
    )
    pixels_h_maps = np.asarray(pixels_h_maps, dtype=np.int64)
    nested = pixels_h_maps if nest else hp.ring2nest(int(nside_h_maps), pixels_h_maps)
    children = nested[:, None] * ratio**2 + np.arange(ratio**2)
    return children if nest else hp.nest2ring(int(nside_final_map), children)


def downgrade_mask(
    mask: np.ndarray,
    projection_pixel: str,
    shape_final_map: tuple,
    shape_h_maps: tuple,
    nest: bool = False,
) -> np.ndarray:
    """
    Mask of the $h$-map grid selecting the pixels that contain at least one observed
    output pixel.

    Equivalent to ``mask.downgrade(ratio) != 0`` for a CAR ``enmap.ndmap``.

    Parameters
    ----------
    mask : np.ndarray
        Mask of the full output map (non-zero for observed pixels).
    projection_pixel : str
        'car' or 'healpix'.
    shape_final_map : tuple
        Shape of the full output map.
    shape_h_maps : tuple
        Shape of the full $h$-map grid.
    nest : bool, optional
        HEALPix ordering: NESTED if True, RING otherwise (default).

    Returns
    -------
    np.ndarray
        Boolean mask of the flattened $h$-map grid.
    """
    observed = np.flatnonzero(np.asarray(mask).ravel())
    mask_h_maps = np.zeros(int(np.prod(shape_h_maps)), dtype=bool)
    mask_h_maps[get_parent_pixels(projection_pixel, observed, shape_final_map, shape_h_maps, nest=nest)] = True
    return mask_h_maps


def get_chunks_healpix(
    mask_h_maps: np.ndarray,
    number_chunks: int = 10,
    nest: bool = False,
) -> list[np.ndarray]:
    """
    Split the observed pixels of a HEALPix $h$-map grid into chunks of pixels consecutive
    in NESTED ordering.

    Pixels consecutive in NESTED ordering form compact regions of the sphere, and their
    children at a finer resolution are consecutive in NESTED ordering as well.

    Parameters
    ----------
    mask_h_maps : np.ndarray
        Boolean mask of the full $h$-map grid.
    number_chunks : int, optional
        Number of chunks. Default is 10.
    nest : bool, optional
        Ordering of ``mask_h_maps`` and of the returned pixels: NESTED if True, RING
        otherwise (default).

    Returns
    -------
    list[np.ndarray]
        For each chunk, the indices of its $h$-map pixels in the full grid, in the
        ordering of the map. The chunks hold the same number of pixels (up to one).
    """
    mask_h_maps = np.asarray(mask_h_maps).ravel() != 0
    nside_h_maps = hp.npix2nside(mask_h_maps.size)
    pixels = np.flatnonzero(mask_h_maps)
    nested = pixels if nest else hp.ring2nest(nside_h_maps, pixels)
    pixels = pixels[np.argsort(nested, kind="stable")]
    return [pixels[start:stop] for start, stop in get_chunks_from_shape((pixels.size,), number_chunks)]


def get_pixels_chunk(
    projection_pixel: str,
    chunk: np.ndarray,
    shape_final_map: tuple,
    shape_h_maps: tuple,
    nest: bool = False,
) -> np.ndarray:
    """
    Indices, in the full output map, of the output pixels covered by a chunk of the
    $h$-map grid.

    Parameters
    ----------
    projection_pixel : str
        'car' or 'healpix'.
    chunk : np.ndarray
        CAR: block ``(start_x, stop_x, start_y, stop_y)`` of the $h$-map grid, as given by
        :func:`get_chunks_from_shape`. HEALPix: $h$-map pixels of the chunk, as given by
        :func:`get_chunks_healpix`.
    shape_final_map : tuple
        Shape of the full output map.
    shape_h_maps : tuple
        Shape of the full $h$-map grid.
    nest : bool, optional
        HEALPix ordering: NESTED if True, RING otherwise (default).

    Returns
    -------
    np.ndarray
        Output pixel indices (flattened full map): row-major order of the output block for
        CAR, children of each $h$-map pixel for HEALPix.
    """
    if projection_pixel == "car":
        chunk_final_map = get_chunks_final_map_from_downgraded_chunks(
            shape_final_map, shape_h_maps, [chunk]
        )[0]
        rows = np.arange(chunk_final_map[0], chunk_final_map[1], dtype=np.int64)
        columns = np.arange(chunk_final_map[2], chunk_final_map[3], dtype=np.int64)
        return (rows[:, None] * int(shape_final_map[1]) + columns[None, :]).ravel()
    return get_children_pixels_healpix(
        chunk,
        hp.npix2nside(int(np.prod(shape_final_map))),
        hp.npix2nside(int(np.prod(shape_h_maps))),
        nest=nest,
    ).ravel()


def get_projector_chunk(
    projection_pixel: str,
    pixels_chunk: np.ndarray,
    observed_pixels_final_map: np.ndarray,
    observed_pixels_h_maps: np.ndarray,
    shape_final_map: tuple,
    shape_h_maps: tuple,
    nest: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Index bookkeeping of a chunk, for compressed output maps and compressed $h$-maps.

    Parameters
    ----------
    projection_pixel : str
        'car' or 'healpix'.
    pixels_chunk : np.ndarray
        Output pixels of the chunk in the full output map (:func:`get_pixels_chunk`),
        observed or not.
    observed_pixels_final_map : np.ndarray
        Sorted indices, in the full output map, of the observed output pixels: the pixels
        held by the compressed output maps (``np.flatnonzero(mask)``).
    observed_pixels_h_maps : np.ndarray
        Sorted indices, in the full $h$-map grid, of the pixels held by the compressed
        $h$-maps.
    shape_final_map : tuple
        Shape of the full output map.
    shape_h_maps : tuple
        Shape of the full $h$-map grid.
    nest : bool, optional
        HEALPix ordering: NESTED if True, RING otherwise (default).

    Returns
    -------
    slice_1d_final_map : np.ndarray
        Positions, in the compressed output maps, of the observed output pixels of the
        chunk, in increasing order.
    slice_1d_hmap : np.ndarray
        Positions, in the compressed $h$-maps, of the $h$-map pixels containing them, in
        increasing order.
    projector_h_n : np.ndarray
        For each pixel of ``slice_1d_final_map``, the position of its parent in
        ``slice_1d_hmap``: ``h_n[..., slice_1d_hmap][..., projector_h_n]`` are the
        $h$-maps of the output pixels of the chunk.
    """
    pixels_chunk = np.asarray(pixels_chunk, dtype=np.int64)
    position = np.searchsorted(observed_pixels_final_map, pixels_chunk)
    position_clipped = np.minimum(position, observed_pixels_final_map.size - 1)
    is_observed = (position < observed_pixels_final_map.size) & (
        observed_pixels_final_map[position_clipped] == pixels_chunk
    )
    order = np.argsort(position[is_observed], kind="stable")
    slice_1d_final_map = position[is_observed][order]
    pixels = pixels_chunk[is_observed][order]

    parents = get_parent_pixels(projection_pixel, pixels, shape_final_map, shape_h_maps, nest=nest)
    columns = np.searchsorted(observed_pixels_h_maps, parents)
    columns_clipped = np.minimum(columns, observed_pixels_h_maps.size - 1)
    assert np.all(columns < observed_pixels_h_maps.size) and np.all(
        observed_pixels_h_maps[columns_clipped] == parents
    ), "Some observed output pixels fall in h-map pixels that are not held by the h-maps"

    slice_1d_hmap, projector_h_n = np.unique(columns, return_inverse=True)
    return slice_1d_final_map, slice_1d_hmap, projector_h_n.ravel()


def upgrade_h_maps(
    h_n_spin_dict: dict | Spin_maps,
    projector_h_n: np.ndarray,
) -> Spin_maps:
    """
    $h$-maps at the output resolution, through indexation of coarser $h$-maps.

    Useful to run :meth:`smarties.mapmaking.FrameworkSystematics.compute_total_maps` with
    $h$-maps computed at a coarser resolution, when the $h$-maps at the output resolution
    fit in memory; the mapmaking per chunk does the same operation chunk by chunk.

    Parameters
    ----------
    h_n_spin_dict : dict or Spin_maps
        Compressed $h$-maps, ``spin -> (n_det, n_h_map_pixels)``; a ``(n_det, 1)`` map is
        kept as is.
    projector_h_n : np.ndarray
        For each output pixel, the position of its parent in the compressed $h$-maps, e.g.
        ``get_projector_map_resolution(...)[mask_final_map.ravel() != 0]`` or
        ``np.searchsorted(observed_pixels_h_maps, get_parent_pixels(...))``.

    Returns
    -------
    Spin_maps
        $h$-maps of shape ``(n_det, projector_h_n.size)``.
    """
    upgraded = Spin_maps()
    for spin, value in h_n_spin_dict.items():
        value = np.asarray(value)
        upgraded[spin] = value if value.shape[-1] == 1 else value[..., projector_h_n]
    return upgraded
