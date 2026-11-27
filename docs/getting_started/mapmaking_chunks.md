# Mapmaking per Chunk

[`FrameworkSystematicsChunks`][smarties.mapmaking.FrameworkSystematicsChunks] extends
[`FrameworkSystematics`][smarties.mapmaking.FrameworkSystematics] with
[`compute_total_maps_chunks`][smarties.mapmaking.FrameworkSystematicsChunks.compute_total_maps_chunks], to solve for the same mapmaking equation chunk of pixels by chunk of pixels, on HEALPix or CAR maps. Use it for large maps or large number of detectors on the focal plane:
- **adaptive resolution**: the $h$-maps can be given at a coarser resolution than the output
  maps. Each output pixel uses the $h$-maps of the coarse pixel that contains it, and the
  $h$-maps are never stored at the output resolution;
- **detector-wise interpolation**: the provided detectors' $h$-maps are used to interpolate the $h$-maps of detectors not provided;
- **systematics from templates**: the `(n_det, npix)` systematics maps are built chunk by chunk from per-detector parameters and sky templates, so that the full `(n_det, npix)` systematics maps do not need to be stored;
- **bounded memory**: only the arrays of one chunk are computed at a time.

The output maps are similar to those of [`compute_total_maps`][smarties.mapmaking.FrameworkSystematics.compute_total_maps] with the $h$-maps of all the detectors (interpolated ones included) computed at the output resolution, whatever the number of chunks.

The snippets of this page read the $h$-maps shipped in `tests/data` (litebird_sim $h$-maps on
HEALPix maps, sotodlib $h$-maps on CAR maps); run them from the root of the repository.

## Inputs

| Argument | Content |
|----------|---------|
| `geometry_dictionary` | `"projection_pixel"`: `"healpix"` or `"car"`; for HEALPix, `"nside"` of the output maps and optionally `"nest": True` (NESTED ordering of the output maps and $h$-maps, RING by default); for CAR, `"wcs_car"` and `"shape_car"` (default `map_shape`) of the output maps |
| `h_n_spin_dict` | $h$-maps `n -> (n_det, npix)`, spins $n \ge 0$, normalised ($\sum_d h^d_0 = 1$), on the pixels of the $h$-map grid that contain at least one observed output pixel, in increasing order; `shape_fullsky` (and `wcs_car` for CAR) give the full $h$-map grid, corresponding to the full sky for HEALPix maps and the full field of view for CAR maps |
| `mask` | full output map, non-zero on the observed pixels |
| `spin_sky_maps` | `spin -> (npix,)` on the observed output pixels |
| `spin_systematics_maps` | `spin -> (n_det, npix)` on the observed output pixels, optional |
| `polar_angle`, `polar_efficiency_coeff` | `(n_det,)`, all the detectors (known ones first with interpolation) |
| `number_chunks` | see [Chunks](#chunks) |

The readers of [`smarties.io`][smarties.io] return the $h$-maps in the wanted format when given the mask to apply on the $h$-map grid. The output resolution must be an integer multiple of the $h$-map resolution: CAR shapes multiple along each axis, HEALPix `nside` multiple by a power of 2 (including the same resolution).

## Example on HEALPix Maps

The $h$-maps of `tests/data/lbs_examples` are at $N_{\rm side} = 64$; the output maps are at
$N_{\rm side} = 128$, on a cap of the sky.
[`downgrade_mask`][smarties.utils.resolution.downgrade_mask] gives the pixels of the $h$-map grid
containing observed output pixels, on which the $h$-maps are read:

```python
import glob

import healpy as hp
import numpy as np

from smarties import (
    FrameworkSystematicsChunks,
    Spin_maps,
    create_pointing_spin_leakage_map_BICEP,
    read_lbs_h_maps,
)
from smarties.utils.resolution import downgrade_mask

files = sorted(name[: -len(".h5")] for name in glob.glob("tests/data/lbs_examples/h_maps_det_*.h5"))
angles = np.deg2rad([float(name.split("_")[-1]) for name in files])

nside_h, nside, lmax = 64, 128, 256
theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
mask = theta < np.deg2rad(60)  # observed output pixels, RING ordering
mask_h = downgrade_mask(mask, "healpix", mask.shape, (hp.nside2npix(nside_h),))
h_maps, attributes = read_lbs_h_maps(
    files, prefix="", list_spin=[np.array([n, 0]) for n in range(0, 5)], mask=mask_h
)
print(mask.sum(), h_maps[0].shape, h_maps.shape_fullsky)  # 48896 (8, 12416) 49152

framework = FrameworkSystematicsChunks(map_shape=mask.shape, nstokes=3, lmax=lmax)
sky_full = framework.get_CMB_spin_sky_maps(nside=nside, fwhm=60.0, seed=0)  # FWHM in arcmin
sky = Spin_maps.from_dictionary({spin: value[mask] for spin, value in sky_full.items()})

rng = np.random.default_rng(1)
delta_x, delta_y = np.deg2rad(rng.normal(0, 1 / 60, (2, len(files))))  # pointing offsets
leakage = create_pointing_spin_leakage_map_BICEP(sky_full[0].real, delta_x, delta_y, lmax=lmax, mask=mask)

geometry = {"projection_pixel": "healpix", "nside": nside}
maps, inverse = framework.compute_total_maps_chunks(
    geometry, h_maps, sky, leakage, mask=mask, polar_angle=angles,
    number_chunks=8, return_Q_U=True, return_inverse_mapmaking_matrix=True,
)
print(maps.shape, inverse.shape)  # (3, 48896) (48896, 3, 3)
```

The same maps with
[`compute_total_maps`][smarties.mapmaking.FrameworkSystematics.compute_total_maps], the
$h$-maps being upgraded to the output resolution beforehand
([`upgrade_h_maps`][smarties.utils.resolution.upgrade_h_maps]):

```python
from smarties import FrameworkSystematics
from smarties.utils.resolution import get_parent_pixels, upgrade_h_maps

parents = get_parent_pixels("healpix", np.flatnonzero(mask), mask.shape, mask_h.shape)
projector = np.searchsorted(np.flatnonzero(mask_h), parents)  # column of h_maps of each output pixel
reference = FrameworkSystematics(map_shape=mask.shape, nstokes=3, lmax=lmax).compute_total_maps(
    upgrade_h_maps(h_maps, projector), sky, leakage, mask=mask, polar_angle=angles, return_Q_U=True,
)
print(np.abs(maps - reference).max())  # 0.0
```

## Chunks

`number_chunks` sets how the observed pixels are split:

- CAR: a pair `(n_0, n_1)` of numbers of chunks along each axis of the $h$-map grid, or a scalar
  `n`, giving `int(sqrt(n)) + 1` chunks along each axis;
- HEALPix: the number of chunks of observed $h$-map pixels, consecutive in NESTED ordering
  (compact regions of the sphere).

A chunk holds the $h$-map pixels of its region and all the output pixels they contain. The
arrays of one chunk only — $h$-maps at the output resolution, data vector, mapmaking matrix —
are computed at a time; more chunks reduce the memory, not the result. The progress is
logged chunk by chunk at the `INFO` level (see [Logs](logs.md)).

## Reusing the Inverse Mapmaking Matrix

With `return_inverse_mapmaking_matrix=True`, the inverse mapmaking matrix is returned at the
output resolution, `(npix, nstokes, nstokes)`, each output pixel holding the matrix of its
$h$-map pixel. It only depends on the $h$-maps (interpolated ones included), the polarisation
angles and efficiencies: pass it to the following runs to skip its computation.

```python
for scale in (0.5, 2.0):
    scaled = Spin_maps.from_dictionary({spin: scale * value for spin, value in leakage.items()})
    other = framework.compute_total_maps_chunks(
        geometry, h_maps, sky, scaled, mask=mask, polar_angle=angles,
        inverse_mapmaking_matrix=inverse, return_Q_U=True,
    )
```

Its condition number selects the well-conditioned pixels, see
[Condition number](formalism.md#condition-number).

## Systematics from Templates

With `build_systematics_maps_from_templates=True`, the systematics maps of each chunk are built
from `dict_utils_template_systematics_maps`:

| Key | Content |
|-----|---------|
| `"template_spin_systematics_maps"` | [`Spin_maps`][smarties.hn.Spin_maps] of the templates, `spin -> (npix,)` on the observed output pixels |
| `"parameters_template_systematics"` | `dict` of the detector parameters, `spin -> (n_det,)` |

The systematics map of spin $k$ of detector $d$ is `parameters[k][d] * templates[k]`, added to
`spin_systematics_maps` if those are given too. The spins $k \ge 0$ are enough, the negative
ones being the conjugates. The `create_dict_template_*` functions of
[Systematics](systematics.md#templates) return these two dictionaries:

```python
from smarties import create_dict_template_pointing_BICEP

parameters, templates = create_dict_template_pointing_BICEP(
    sky_full[0].real, delta_x, delta_y, lmax=lmax, mask=mask
)
maps_from_templates = framework.compute_total_maps_chunks(
    geometry, h_maps, sky, mask=mask, polar_angle=angles, number_chunks=8, return_Q_U=True,
    build_systematics_maps_from_templates=True,
    dict_utils_template_systematics_maps={
        "template_spin_systematics_maps": templates,  # spins -1, 1: (npix,)
        "parameters_template_systematics": parameters,  # spins -1, 1: (n_det,)
    },
)
print(np.abs(maps_from_templates - maps).max())  # 0.0: same as with the systematics maps
```

To combine several systematics, merge their dictionaries; for templates of the same spin,
stack them, `(n_templates, n_det)` for the parameters and `(n_templates, npix)` for the
templates: the systematics map is then $\sum_t$ `parameters[k][t, d] * templates[k][t]`.

## Detector-wise Interpolation

With `perform_det_interpolation=True`, the $h$-maps given are those of the *known* detectors,
and the $h$-maps of other detectors are interpolated on the focal plane, chunk by chunk:

1. the hits ($h^d_0$ times `h_maps.total_hits_map`) and the averages $h^d_n / h^d_0$ of the known
   detectors are interpolated linearly at the positions of the other detectors, from at least
   three known detectors, the values outside their convex hull being those of the nearest one
   (with a warning, see [Logs](logs.md));
2. the $h$-maps of all the detectors are normalised again,
   $h^d_0 = \mathrm{hits}_d / \sum_{d'} \mathrm{hits}_{d'}$ and $h^d_n = h^d_0 \times$average$^d_n$.

The detectors are ordered as the known ones (in the order of the $h$-maps), then the
interpolated ones: `polar_angle`, `polar_efficiency_coeff` and the systematics maps follow this
order. The focal plane is described by `dict_utils_interpolation_det`:

| Key | Content |
|-----|---------|
| `"dict_positions_detector_known"` | positions of the known detectors, `key -> (n_known,)`, in the order of the $h$-maps |
| `"dict_positions_detector_unknown"` | positions of the detectors to interpolate, `key -> (n_unknown,)` |
| `"dict_key_positions"` | `{"position_x_key": ..., "position_y_key": ...}`, the keys of the two coordinates |
| `"reweight_by_hits"` | `True` for normalised $h$-maps (read with `reweight_by_hits=True`, the default of the readers, which also set `total_hits_map`); `False` for $h$-maps holding the hits and the averages (read with `reweight_by_hits=False`) |
| `"wafer_list_detector_known"`, `"wafer_list_detector_unknown"` | optional, wafer of each detector: the interpolation is done wafer by wafer, from the known detectors of the same wafer |

The [CAR example](#car-maps-a-simons-observatory-example) below interpolates the $h$-maps of a
detector pair from the focal-plane positions, wafers and polarisation angles stored in the
sotodlib files, and compares them with the actual ones with
[`get_interpolated_h_maps_chunk`][smarties.mapmaking.FrameworkSystematicsChunks.get_interpolated_h_maps_chunk].

## CAR Maps: an example with Simons Observatory LAT-like $h$-maps

`tests/data/sotodlib_examples` holds the sotodlib $h$-maps of 10 detectors (5 pairs `A`/`B`) of
the wafer `w16` of a LAT-like scan, at $12'$ on a $115 \times 460$ CAR grid (see
[$h$-maps](h_maps.md#toast-sotodlib-h-maps)). The output maps are three times finer ($4'$), and
the $h$-maps of the pair `p200` are interpolated from those of the 4 other pairs:

```python
import glob

import numpy as np
from pixell import enmap

from smarties import (
    FrameworkSystematicsChunks,
    build_mask_from_toast_h_maps,
    read_toast_h_maps,
    transform_array_maps_into_spin_maps,
)

files = sorted(glob.glob("tests/data/sotodlib_examples/h_n_w16_*_LAT_f090_[AB].hdf"))
known_files = [name for name in files if "_p200_" not in name]  # 8 detectors with h-maps
new_files = [name for name in files if "_p200_" in name]  # 2 detectors to interpolate

mask_h, shape_h = build_mask_from_toast_h_maps(known_files, return_original_shape=True)
h_maps, attributes = read_toast_h_maps(known_files, mask=mask_h)  # attributes: positions, angles...

# Output geometry, 3 times finer along each axis, and its observed pixels
shape, wcs = enmap.upgrade_geometry(shape_h, h_maps.wcs_car, 3)
mask = enmap.upgrade(enmap.ndmap(np.asarray(mask_h).reshape(shape_h), h_maps.wcs_car), 3)
print(shape_h, tuple(int(s) for s in shape), int((mask != 0).sum()))  # (115, 460) (345, 1380) 261927

# A Gaussian random I, Q, U sky on the output geometry
ell = np.arange(3001)
cl = np.zeros(ell.size)
cl[2:] = 1e3 / ell[2:] ** 2
power_spectra = np.zeros((3, 3, ell.size))
power_spectra[0, 0], power_spectra[1, 1], power_spectra[2, 2] = cl, cl / 100, cl / 1000  # TT, EE, BB
iqu = enmap.rand_map((3,) + tuple(shape), wcs, power_spectra, seed=0)
sky = transform_array_maps_into_spin_maps(iqu[:, mask != 0])

# Detectors to interpolate: focal-plane positions (mm), wafers and polarisation angles (radians),
# usually given by the focal-plane model of the instrument, here read from the files of p200
_, attributes_new = read_toast_h_maps(new_files, list_spin=[0], mask=mask_h)
wafer = lambda detectors: np.array([name.split("_")[0] for name in detectors])  # "w16_p000_LAT_f090_A" -> "w16"
dict_utils_interpolation_det = {
    "dict_positions_detector_known": {"x": attributes["wafer_x_mm"], "y": attributes["wafer_y_mm"]},
    "dict_positions_detector_unknown": {"x": attributes_new["wafer_x_mm"], "y": attributes_new["wafer_y_mm"]},
    "dict_key_positions": {"position_x_key": "x", "position_y_key": "y"},
    "reweight_by_hits": True,
    "wafer_list_detector_known": wafer(attributes["det"]),
    "wafer_list_detector_unknown": wafer(attributes_new["det"]),
}

framework = FrameworkSystematicsChunks(map_shape=shape, nstokes=3, lmax=3000)
maps, inverse = framework.compute_total_maps_chunks(
    {"projection_pixel": "car", "shape_car": shape, "wcs_car": wcs},
    h_maps, sky, mask=mask,
    polar_angle=np.concatenate([attributes["pol_angle_rad"], attributes_new["pol_angle_rad"]]),  # known first
    perform_det_interpolation=True, dict_utils_interpolation_det=dict_utils_interpolation_det,
    number_chunks=(4, 8), return_Q_U=True, return_inverse_mapmaking_matrix=True,
)
well_conditioned = np.linalg.cond(inverse) < 9
print(np.abs(maps.real - iqu[:, mask != 0])[:, well_conditioned].max())  # 5.7e-14: no systematics, the input sky

output = enmap.zeros((3,) + tuple(shape), wcs)  # I, Q, U on the full output geometry
output[:, mask != 0] = maps.real
```

The interpolated $h$-maps of the pair `p200`, against its actual $h$-maps:

```python
pixels = np.arange(h_maps[0].shape[1])  # all the columns of the h-maps
h_maps_interpolated = framework.get_interpolated_h_maps_chunk(h_maps, pixels, dict_utils_interpolation_det)
h_maps_actual, _ = read_toast_h_maps(files, mask=mask_h)  # the 10 detectors, p200 last
print(h_maps_interpolated[0].shape)  # (10, 29103): the 8 known detectors, then the 2 interpolated ones
print(np.median(np.abs(h_maps_interpolated[0][8:] - h_maps_actual[0][8:])))  # 1.7e-3, for h_0 ~ 1/10
```
