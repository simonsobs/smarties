# Mapmaking

[`FrameworkSystematics`][smarties.mapmaking.FrameworkSystematics] solves the spin mapmaking
equation of the [Spin formalism](formalism.md) page. It has two entry points:

- [`compute_total_maps`][smarties.mapmaking.FrameworkSystematics.compute_total_maps] — all the
  observed pixels at once, HEALPix or CAR;
- [`compute_total_maps_chunks`][smarties.mapmaking.FrameworkSystematics.compute_total_maps_chunks]
  — mapmaking per chunk of CAR pixels, with on-the-fly adaptive resolution of the $h$-maps,
  systematics built from templates and detector-wise interpolation of the $h$-maps.

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
h_maps = read_lbs_h_maps(
    files, prefix="", list_spin=[np.array([n, 0]) for n in range(1, 5)],
    mask=mask, list_weights=np.ones(len(files)),
)
npix, lmax = mask.size, 128
sky = framework.get_CMB_spin_sky_maps(nside=nside, fwhm=np.deg2rad(2.0), seed=0)


rng = np.random.default_rng(4)
delta_x, delta_y = np.deg2rad(rng.normal(0, 1 / 60, (2, len(files))))
leakage = create_pointing_spin_leakage_map_BICEP(sky[0], delta_x, delta_y, lmax=lmax)

# compute_total_maps completes the dictionaries it receives with zero maps for the spins they
# lack: new containers (holding the same arrays) keep sky and leakage unchanged for later calls
maps = framework.compute_total_maps(h_maps, sky, leakage, polar_angle=angles)
print(sorted(maps.keys()))  # [-2, 0, 2]: I and Q -+ iU
```

The output spin maps are $I$ (spin 0) and $Q \pm iU$ (spins $\pm 2$); with `return_Q_U=True`
the mapmaker returns an array of shape `(nstokes, npix)` holding I, Q, U.

### Masks

`mask` selects the observed pixels. The inputs can be given in two ways:

=== "Inputs restricted to the mask (default)"

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

=== "Full inputs, masked by the mapmaker (not recommended)"

    ```python
    theta, _ = hp.pix2ang(hp.npix2nside(npix), np.arange(npix))
    cap = (theta < np.pi / 3).astype(int)
    out = framework.compute_total_maps(
        h_maps, sky, leakage, mask=cap, polar_angle=angles, return_Q_U=True, mask_input=True,
    )
    print(out.shape)  # (3, number of observed pixels)
    ```

    `mask_input=True` copies every input map; avoid it for large focal planes.

### Reusing the Inverse Mapmaking Matrix

The mapmaking matrix only depends on the $h$-maps, the polarisation angles and the
efficiencies. Compute its inverse once and pass it to the following runs:

```python
maps, inverse = framework.compute_total_maps(
    h_maps, sky, leakage, polar_angle=angles, return_inverse_mapmaking_matrix=True,
)
print(inverse.shape)  # (npix, 3, 3)

for scale in (0.5, 2.0):
    scaled = Spin_maps.from_dictionary({k: scale * v for k, v in leakage.items()})
    other = framework.compute_total_maps(
        h_maps, sky, scaled, polar_angle=angles, inverse_mapmaking_matrix=inverse
    )

condition_number = np.linalg.cond(inverse)  # select the well-conditioned pixels
print(condition_number.min(), condition_number.max())
```

### Other Options

| Argument | Effect |
|----------|--------|
| `polar_angle` | detector polarisation angles $\gamma_d$ (radians, `(n_det,)`) |
| `polar_efficiency_coeff` | polarisation efficiencies assumed by the mapmaker, `(n_det,)` |
| `projector_h_n` | integer indices mapping each output pixel to an $h$-map pixel, for $h$-maps at another resolution |
| `geometry_dictionary={"wcs_car": wcs}` | return `enmap.ndmap` outputs for CAR maps |
| `single_precision` | `complex64` computations |
| `optimize_computations` | compute only the non-negative spins and conjugate (default `True`) |

## Mapmaking per Chunk

`compute_total_maps_chunks` processes the observed pixels of a CAR map chunk by chunk, which
bounds the memory, and offers three features of its own.

**On-the-fly adaptive resolution.** The $h$-maps encompass most information at large scales: 
they can be computed and stored at a coarser resolution than the output maps (if necessary). 
During the mapmaking, each output pixel uses the $h$-maps of the coarse pixel containing it, 
and the mapmaking matrix is inverted at the coarse resolution, so that the $h$-maps are never 
stored at the output resolution. 

**Systematics from templates.** Most systematics factorise into a per-detector parameter times
a sky template (see [Systematics](systematics.md#templates)); the systematics maps can be built pixel chunks by pixel chunks, without storing `(n_det, npix)` arrays.

**Detector-wise interpolation.** The $h$-maps of all the detectors of a focal plane can be
interpolated, from the $h$-maps of a subset of detectors and the positions of the
detectors on the focal plane.

The inputs are:

- `geometry_dictionary` with `"projection_pixel": "car"`, and `"shape_car"` and `"wcs_car"`, the
  shape and WCS of the **output** map;
- `h_maps` carrying the geometry of the $h$-maps in `h_maps.shape_fullsky` and
  `h_maps.wcs_car` (set by [`read_toast_h_maps`][smarties.io.read_toast_h_maps]), with the
  negative spins stored, on the coarse pixels containing an observed output pixel;
- `mask`, the observed pixels on the output grid, as an `enmap.ndmap`;
- `number_chunks`, a tuple `(n_y, n_x)` of chunks along each axis of the $h$-map grid, dividing
  the $h$-map shape on both axes.

As an example, the SO $h$-maps of `tests/data/sotodlib_examples` ($345 \times 1380$ pixels of
$4'$) are used for an output map twice as fine:

```python
from pixell import enmap

from smarties import build_mask_from_toast_h_maps, read_toast_h_maps

pairs = sorted(p[: -len("_A_cos_0.hdf")] for p in glob.glob("tests/data/sotodlib_examples/h_n_*_A_cos_0.hdf"))
mask_h, shape_h = build_mask_from_toast_h_maps(
    pairs, format_file="hdf", spin_to_read=2, projection_pixel="car", return_original_shape=True
)
h_maps_car = read_toast_h_maps(pairs, list_spin=[0, 1, 2, 3, 4], format_file="hdf", mask=mask_h, projection_pixel="car")

shape, wcs = enmap.upgrade_geometry(tuple(shape_h), h_maps_car.wcs_car, 2)  # output grid
shape = tuple(int(s) for s in shape)
mask_output = enmap.ndmap(np.repeat(np.repeat(mask_h.reshape(shape_h), 2, axis=0), 2, axis=1), wcs)
n_observed = int((mask_output != 0).sum())

I, Q, U = np.random.default_rng(5).standard_normal((3, n_observed))
sky_car = Spin_maps.from_dictionary({0: I, 2: (Q + 1j * U) / 2, -2: (Q - 1j * U) / 2})

framework_car = FrameworkSystematics(map_shape=shape, nstokes=3, lmax=1000)
maps_car = framework_car.compute_total_maps_chunks(
    {"projection_pixel": "car", "shape_car": shape, "wcs_car": wcs},
    h_maps_car, sky_car, mask=mask_output, number_chunks=(5, 4), return_Q_U=True,
)
print(shape_h, "->", shape, maps_car.shape)  # (345, 1380) -> (690, 2760) (3, n_observed)
```

### Templates

With `build_systematics_maps_from_templates=True`, the systematics are rebuilt chunk by chunk as
parameter × template:

```python
dict_utils_template_systematics_maps = {
    "template_spin_systematics_maps": templates,  # Spin_maps, spin -> (npix,)
    "parameters_template_systematics": parameters,  # dict, spin -> (n_det,)
}
```

The spins of the templates must also be present in `spin_systematics_maps` (zero arrays such as
`np.zeros((1, 1))` are enough).

### Detector-wise Interpolation

With `perform_det_interpolation=True`, the $h$-maps of the detectors without simulated $h$-maps
are interpolated from those of the known detectors
([`perform_interpolation_scipy`][smarties.utils.interpolation.perform_interpolation_scipy],
linear interpolation on the focal plane with a nearest-detector fallback). The focal-plane
positions are given in `dict_utils_interpolation_det`:

```python
dict_utils_interpolation_det = {
    "dict_positions_detector_known": {"wafer_x_mm": x_known, "wafer_y_mm": y_known},
    "dict_positions_detector_unkown": {"wafer_x_mm": x_new, "wafer_y_mm": y_new},
    "dict_key_positions": {"position_x_key": "wafer_x_mm", "position_y_key": "wafer_y_mm"},
    "list_weights": weights,  # (n_known, npix)
}
```

`polar_angle` and the systematics maps then cover the known detectors first, followed by the
interpolated ones.

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
