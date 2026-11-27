# Mapmaking

Two classes solve the spin mapmaking equation of the [Spin formalism](formalism.md) page:

- [`FrameworkSystematics.compute_total_maps`][smarties.mapmaking.FrameworkSystematics.compute_total_maps]
  — all the observed pixels at once, HEALPix or CAR, with the $h$-maps at the output
  resolution (this page);
- [`FrameworkSystematicsChunks.compute_total_maps_chunks`][smarties.mapmaking.FrameworkSystematicsChunks.compute_total_maps_chunks]
  — mapmaking per chunk of pixels, HEALPix or CAR, with on-the-fly adaptive resolution of the $h$-maps, detector-wise interpolation of the $h$-maps and systematics built from templates (see [Mapmaking per chunk](mapmaking_chunks.md)).

The snippets of this page read the $h$-maps shipped in `tests/data`; run them from the root of
the repository.

## The Framework Object

```python
from smarties import FrameworkSystematics

framework = FrameworkSystematics(
    map_shape=(12 * 64**2,),  # full pixel shape: (npix,) for HEALPix, (ny, nx) for CAR
    nstokes=3,  # 1: I, 2: Q/U, 3: I/Q/U
    lmax=128,  # used to generate CMB skies
    list_spin_output=[0, -2, 2],  # one spin per Stokes parameter, symmetric around 0
)
```

`list_spin_output` must contain `nstokes` spins: `[0, -2, 2]` (default) for I, Q, U, `[-2, 2]`
for polarisation only, `[0]` for intensity only. Their order does not matter.

## Mapmaking on All the Observed Pixels

We use the HEALPix $h$-maps of the LiteBIRD-like scan of `tests/data/lbs_examples`, a sky, and a
differential pointing systematic:

```python
import glob

import healpy as hp
import numpy as np

from smarties import (
    Spin_maps,
    create_pointing_spin_leakage_map_BICEP,
    read_lbs_h_maps,
    transform_array_maps_into_spin_maps,
)
from smarties.io import build_mask_from_lbs_h_maps

files = sorted(path[: -len(".h5")] for path in glob.glob("tests/data/lbs_examples/h_maps_det_*.h5"))
angles = np.deg2rad([float(name.split("_")[-1]) for name in files])
mask = build_mask_from_lbs_h_maps(files)
h_maps, attributes = read_lbs_h_maps(
    files, prefix="", list_spin=[np.array([n, 0]) for n in range(0, 5)], mask=mask,
)
npix = mask.size
lmax = 2 * hp.npix2nside(npix)
sky = framework.get_CMB_spin_sky_maps(nside=hp.npix2nside(npix), fwhm=120.0, seed=0)  # FWHM in arcmin


rng = np.random.default_rng(4)
delta_x, delta_y = np.deg2rad(rng.normal(0, 1 / 60, (2, len(files))))
leakage = create_pointing_spin_leakage_map_BICEP(sky[0], delta_x, delta_y, lmax=lmax)

maps = framework.compute_total_maps(h_maps, sky, leakage, polar_angle=angles)
print(maps.spins)  # [0, 2]: I and Q + iU
```

The output spin maps are $I$ (spin 0) and $Q \pm iU$ (spins $\pm 2$); only the non-negative
spins are stored, `maps[-2]` returning the conjugate of `maps[2]`. With `return_Q_U=True` the
mapmaker returns an array of shape `(nstokes, npix)` holding I, Q, U.

### Masks

`mask` selects the observed pixels of the full map. The $h$-maps, sky and systematics maps are
given on these pixels only, in increasing order of the pixel index:

```python
theta, _ = hp.pix2ang(hp.npix2nside(npix), np.arange(npix))
cap = (theta < np.pi / 3).astype(int)
observed = np.flatnonzero(cap)

cut = lambda maps: Spin_maps.from_dictionary({k: np.asarray(v)[..., observed] for k, v in maps.items()})
out = framework.compute_total_maps(
    cut(h_maps), cut(sky), cut(leakage), mask=cap, polar_angle=angles, return_Q_U=True
)
print(out.shape)  # (3, number of observed pixels)
```

The readers of [`smarties.io`][smarties.io] return the $h$-maps on the pixels of the `mask` they are given.

### Reusing the Inverse Mapmaking Matrix

The mapmaking matrix only depends on the $h$-maps, the polarisation angles and the
efficiencies. Compute its inverse once and pass it to the following runs:

```python
maps, inverse_mapmaking_matrix = framework.compute_total_maps(
    h_maps, sky, leakage, polar_angle=angles, return_inverse_mapmaking_matrix=True,
)
print(inverse_mapmaking_matrix.shape)  # (npix, 3, 3)

for scale in (0.5, 2.0):
    scaled = Spin_maps.from_dictionary({k: scale * v for k, v in leakage.items()})
    other = framework.compute_total_maps(
        h_maps, sky, scaled, polar_angle=angles, inverse_mapmaking_matrix=inverse_mapmaking_matrix
    )

condition_number = np.linalg.cond(inverse_mapmaking_matrix)  # select the well-conditioned pixels
print(condition_number.min(), condition_number.max())
```

### Other Options

| Argument | Effect |
|----------|--------|
| `polar_angle` | detector polarisation angles $\gamma_d$ (radians, `(n_det,)`) |
| `polar_efficiency_coeff` | polarisation efficiencies assumed by the mapmaker, `(n_det,)` |
| `geometry_dictionary={"wcs_car": wcs}` | return `enmap.ndmap` outputs for CAR maps |
| `single_precision` | `complex64` computations |

## Mapmaking per Chunk

[`FrameworkSystematicsChunks`][smarties.mapmaking.FrameworkSystematicsChunks] processes the
observed pixels chunk by chunk, which bounds the memory, with:
- $h$-maps possibly provided at a coarser resolution than the output maps,
- $h$-maps possibly provided for few detectors and interpolated to detectors without provided $h$-maps,
- systematics built on-the-fly from templates instead of being pre-computed. 
It gives the same maps as `compute_total_maps`: see [Mapmaking per chunk](mapmaking_chunks.md).

## Saving the Results

[`save_partial_spin_maps`][smarties.io.save_partial_spin_maps] writes output spin maps as I, Q, U
expanded to the full map (`.npy` keeps only the observed pixels). CAR outputs (obtained with
`geometry_dictionary={"wcs_car": wcs}`) are written with pixell.

```python
from smarties import save_partial_spin_maps

save_partial_spin_maps(maps, nstokes=3, shape_pixels=(npix,), mask_on_full_map=mask,
                       path_output="maps_with_pointing", format_output=".fits")
```

## Power Spectra Convenience Functions

[`smarties.utils.master`][smarties.utils.master] wraps NaMaster (`uncouple_spectra_TT`,
`uncouple_spectra_pol`, `uncouple_spectra_all`, `uncouple_cross_spectra`) and pspy
(`uncouple_spectra_pspy`) to compute mode-decoupled power spectra of the output maps on a
masked sky; `filter_map_ell_cut` applies harmonic high- and low-pass filters.
