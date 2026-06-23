from copy import deepcopy
import toml, yaml
import numpy as np
import healpy as hp
from opt_einsum import contract
from pixell import enmap, enplot, utils

from smarties.utils.tools import get_coupled_spin, get_row_mapmaking_matrix, reweight_hmaps_by_hits
from smarties.sky.cmb import create_CMB_spin_maps
from smarties.hn import Spin_maps, Spin_nm

def get_chunks_from_shape(
        shape_map, 
        number_chunks=10,
    ):
    array_shape_map = np.array(shape_map)
    if len(array_shape_map) == 1:
        assert np.array(number_chunks).size == 1, 'Number of chunks should be a scalar for 1D maps.'

        step_pixels = array_shape_map[0]//number_chunks

        list_to_return = [ (i*step_pixels, (i+1)*step_pixels) for i in range(number_chunks) ] 
        if list_to_return[-1][1] < array_shape_map[0]:
            list_to_return = list_to_return + [(array_shape_map[0]//step_pixels)*step_pixels, array_shape_map[0]]
        return np.int_(list_to_return)
        
    elif len(array_shape_map) == 2:
        assert np.array(number_chunks).size < 3, 'Number of chunks should be a scalar or 2D array for 2D maps.'
        if np.array(number_chunks).size == 1:
            sqrt_number_chunks = int(np.sqrt(number_chunks)) + 1
            number_chunks_x, number_chunks_y = (sqrt_number_chunks, sqrt_number_chunks)
        else:
            number_chunks_x, number_chunks_y = number_chunks

        step_pixels = array_shape_map//np.array([number_chunks_x, number_chunks_y])

        x_chunks = np.arange(number_chunks_x+1)*step_pixels[0]
        y_chunks = np.arange(number_chunks_y+1)*step_pixels[1]

        return np.int_([
            (x_chunks[i], x_chunks[i+1], y_chunks[j], y_chunks[j+1]) 
            for i in range(len(x_chunks)-1) 
            for j in range(len(y_chunks)-1)
        ]
    )

def get_projector_map_resolution(
        wcs_downgraded,
        shape_map_downgraded,
        projection_pixel,
        factor=2,
        boolean_mask=None,
    ):
    assert projection_pixel in ['healpix', 'car'], f'Unknown projection pixel: {projection_pixel}'
    if projection_pixel == 'healpix':
        raise NotImplementedError('Healpix projection is not implemented yet.')
    
    boolean_mask = ... if boolean_mask is None else boolean_mask.ravel()
    npix = np.prod(boolean_mask[boolean_mask].shape)  

    indices_fake_map = enmap.zeros(
        (shape_map_downgraded[0], shape_map_downgraded[1]), 
        wcs=wcs_downgraded,
        dtype=np.int32
    )
    indices_fake_map.ravel()[boolean_mask] = np.arange(npix)

    return indices_fake_map.upgrade(factor=factor).ravel()


def get_chunks_final_map_from_downgraded_chunks(
        shape_final_map,
        shape_map_downgraded,
        list_chunks_previous_map,
):
    assert np.array(shape_final_map).size == np.array(shape_map_downgraded).size, 'Shape of the full map and the previous map should have the same number of dimensions.'

    if np.array(shape_final_map).ndim == 1:
        ratio = np.array(shape_final_map)[0]//np.array(shape_map_downgraded)[0]
        return np.array(list_chunks_previous_map)*ratio
    
    elif np.array(shape_final_map).ndim == 2:
        ratio_x = np.array(shape_final_map)[0]//np.array(shape_map_downgraded)[0]
        ratio_y = np.array(shape_final_map)[1]//np.array(shape_map_downgraded)[1]
        return np.array(list_chunks_previous_map)*np.array([ratio_x, ratio_x, ratio_y, ratio_y])

def get_slice_1d_from_2d(
        shape_map, 
        slice_x, 
        slice_y,
        boolean_mask=None
    ):
    if slice_y is ... or slice_x is ...:
        return slice_x if slice_x is not ... else slice_y
    indices_map = enmap.zeros(shape_map)
    indices_map[:] = np.arange(shape_map[0]*shape_map[1]).reshape(shape_map)

    if boolean_mask is not None:
        npix = np.prod(boolean_mask[boolean_mask].shape)
        indices_map[:] = -1
        indices_map.ravel()[boolean_mask.ravel()] = np.arange(npix)
    indices_output_1d = indices_map[slice_x, slice_y].ravel()
    return np.int_(indices_output_1d[indices_output_1d >= 0])


def ud_grade_hn(h_n_maps, nside_out):
    """
    Change the resolution of the $h_n$ maps to a lower or higher resolution, 
    by averaging or repeating the pixels in the provided output resolution 
    using the `ud_grade` function from HEALPix. 

    Parameters
    ----------
    h_n_maps: Spin_maps
        Spin maps containing the $h_n$ maps, with keys being the spins and values being
        the maps of shape (n_det, n_pix) or (n_det,) for spin=0.
    nside_out: int
        The desired output resolution, given as nside.

    Returns
    -------
    new_h_n: Spin_maps
        A new Spin_maps object containing the $h_n$ maps at the desired resolution,
        with the same spins as the input maps. The maps are of shape (n_det, n_pix) or 
        (n_det,) for spin=0, where n_det is the number of detectors (1 for spin=0) and 
        n_pix is the number of pixels at the output resolution.

    Notes
    -----
    Currently the corresponding operations only work with HEALPix maps, so the input maps must be provided in the HEALPix format. 
    """

    #TODO: Adapt in case h_n maps are not healpix or ful sky
    
    #TODO: Take gradient conjugate for the -spin?
    
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
