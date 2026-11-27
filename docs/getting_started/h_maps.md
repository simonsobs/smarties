# $h$-maps

The $h$-maps are the only information SMARTIES needs about the scanning strategy. For a detector
$d$ and a pixel $p$,

$$
h^d_n(p) = w^d(p)\, \big\langle e^{in\psi_t} \big\rangle_{t \in p},
\qquad \sum_d w^d(p) = 1,
$$

where:
- the average $\langle \cdot \rangle$ runs over the samples of detector $d$ falling in pixel $p$, 
- $w^d$ is the relative weight of the detector (usually its hit count, normalised over the detectors)
- $\psi_t$ is the angle measured in the plane tangent to the celestial sphere at the observed sky position, between the local meridian (i.e., the direction towards the North Pole of the chosen celestial coordinate system) and a fixed focal plane reference direction projected onto the sky.
 
This latter quantity $\psi_t$ encompasses all the time-dependent rotations of the focal plane with respect to the sky, and we refer to it as the parallactic angle, following common usage in CMB analyses. They are computed once by a time-ordered data simulator and reused for any number of sky and systematics realisations; see [Spin formalism](formalism.md) for their role in the mapmaking.

The snippets of this page read the $h$-maps shipped in `tests/data` of the repository; run them
from the root of the repository.

## Which Spins to Produce

Produce every $h_n$ that couples your systematics to the I, Q, U outputs:

- $h_0, h_2, h_4$ for the sky itself and for spin-0 / spin-2 effects,
- $h_1, h_3$ in addition for pointing offsets (spin 1),
- up to $h_{m_{\max}+2}$ for a beam described up to the azimuthal order $m_{\max}$.

See the table in [Which $h$-maps are needed](formalism.md#which-h-maps-are-needed).

## TOAST / sotodlib $h$-maps

[`read_toast_h_maps`][smarties.io.read_toast_h_maps] reads the HDF5 files written by sotodlib,
one per detector, e.g. `h_n_w16_p000_LAT_f090_A.hdf`, with groups `"n,0"` holding the real (`Re`) and imaginary (`Im`) parts of the $h$-map of spin $n$. Without half-wave plate (default, the half-wave plate is not yet implemented), the keys are reduced to the integer spins $n$. `tests/data/sotodlib_examples` contains the CAR $h$-maps of 10 detectors from a wafer (`w16`) of a Simons Observatory LAT-like scan.

| Content of a file | Meaning |
|-------------------|---------|
| group `"0,0"`, dataset `Re` | hit counts of the detector |
| group `"n,0"`, datasets `Re`, `Im` | $\langle \cos n\psi \rangle$ and $\langle \sin n\psi \rangle$ in each pixel |
| attributes `pixelization` (`"WCS"`), `wcs_header`, `wcs_shape` | CAR geometry |
| attributes `pixelization` (`"HEALPIX"`), `nside`, `ordering` | HEALPix geometry |
| other attributes, e.g. `det`, `pol_angle_rad`, `wafer_x_mm`, `wafer_y_mm` | detector metadata |

```python
import glob

import numpy as np

from smarties import build_mask_from_toast_h_maps, read_toast_h_maps

files = sorted(glob.glob("tests/data/sotodlib_examples/h_n_w16_*_LAT_f090_[AB].hdf"))  # one file per detector

# Observed pixels (hits of at least one detector), detectors per pixel, full map shape
mask, detectors_hits, shape = build_mask_from_toast_h_maps(
    files, return_detectors_hits=True, return_original_shape=True
)
h_maps, attributes = read_toast_h_maps(files, list_spin=[0, 1, 2, 3, 4], mask=mask)
print(h_maps.spins, h_maps[2].shape)  # [0, 1, 2, 3, 4] (n_det, n_observed)
print(h_maps.projection_pixel, h_maps.shape_fullsky)  # car (115, 460)
polarisation_angles = attributes["pol_angle_rad"]  # one value per detector, in the order of files
assert np.allclose(h_maps[0].sum(axis=0), 1.0)
```

What the reader does:

- reads `Re + i Im` for each spin $n \ge 0$ of `list_spin`, which must contain 0 (the hits);
  only the spins $n \ge 0$ are stored, $h_{-n} = \overline{h_n}$ being
  [reconstructed when needed](spin_maps.md#negative-spins);
- with `mask`, keeps only the pixels where the mask is non-zero, flattened (HEALPix maps in
  RING ordering);
- with `reweight_by_hits=True` (default), normalises the maps so that $\sum_d h^d_0 = 1$:
  $h^d_n = w_d\,\mathrm{hits}_d \langle e^{in\psi}\rangle_d / \sum_{d'} w_{d'}\,\mathrm{hits}_{d'}$,
  with the weights `list_weights` (default: 1 for every detector);
- records on the returned [`Spin_maps`][smarties.hn.Spin_maps] the attributes `projection_pixel`, `shape_fullsky`, `wcs_car` (set to `None` for HEALPIX projections) and `total_hits_map`
  ($\sum_d w_d\,\mathrm{hits}_d$);
- returns the attributes of the files, one array per attribute, in the order of the files
  (`return_attributes=False` to get the $h$-maps only).

!!! tip "Check the normalisation"
    The mapmaker requires $\sum_d h^d_0 = 1$ in every pixel: check
    `h_maps[0].sum(axis=0)` after reading.

Options worth knowing:

| Argument | Effect |
|----------|--------|
| `format_file` | extension of the files, `"hdf"` (default), `"h5"`, `"hdf5"`, `"H5"` |
| `reweight_by_hits=False` | $h^d_n = w_d \langle e^{in\psi}\rangle_d$ and $h^d_0 = w_d\,\mathrm{hits}_d$, without normalisation |
| `single_precision=True` | `complex64` maps, halves the memory |
| `return_only_positive_spins=False` | also store the negative spins, otherwise the negative spins are computed on-the-fly from the positive ones |

## litebird_sim $h$-maps

[`read_lbs_h_maps`][smarties.io.read_lbs_h_maps] reads the HDF5 files written by litebird_sim:
one file per detector, with groups `"n,m"` holding the real (`Re`) and imaginary (`Im`) parts
of the $h$-map of spin $n$ and half-wave-plate spin $m$. Without half-wave plate ($m = 0$), the
keys are reduced to the integer spins $n$. `tests/data/lbs_examples` contains the HEALPix
$h$-maps of 8 detectors of a LiteBIRD-like scan:

```python
from smarties import read_lbs_h_maps
from smarties.io import build_mask_from_lbs_h_maps

files = sorted(path[: -len(".h5")] for path in glob.glob("tests/data/lbs_examples/h_maps_det_*.h5"))
mask_lbs = build_mask_from_lbs_h_maps(files)
h_maps_lbs, attributes_lbs = read_lbs_h_maps(
    files,
    prefix="",  # the names above are complete paths
    list_spin=[np.array([n, 0]) for n in range(0, 5)],  # (0, 0): the hits, needed to normalise
    mask=mask_lbs,
)
print(sorted(int(spin) for spin in h_maps_lbs), h_maps_lbs[2].shape)  # [0, 1, 2, 3, 4] (8 detectors, 49152 pixels)
print(attributes_lbs["det"][0])  # 016_000_XX_334_000
```

The normalisation, the weights `list_weights`, the metadata and the attributes are the same as
for the sotodlib reader above.

## Producing $h$-maps

- **TOAST / sotodlib**: the Simons Observatory simulation and mapmaking pipelines can accumulate
  the $h$-maps of each detector while scanning, written in the format above.
- **litebird_sim**: the $h$-maps are produced by the [simulation framework of LiteBIRD](https://github.com/litebird/litebird_sim/blob/master/notebooks/litebird_sim_smarties.ipynb), in the format read by `read_lbs_h_maps`.
- **Planck**: the Planck polmoments are the $h$-maps of the Planck scan; the
  [smarties4planck](https://github.com/camilledemay/smarties4planck) package reads them and converts them to the SMARTIES conventions.

Any other simulator can be used: store, for each detector, the hit counts and
$\langle e^{in\psi}\rangle$ per pixel, and build a [`Spin_maps`][smarties.hn.Spin_maps] with
`h_maps[0] = hits` and `h_maps[n] = averages[n]` (`(n_det, npix)` arrays);
[`reweight_hmaps_by_hits(h_maps, in_place=True)`][smarties.utils.tools.reweight_hmaps_by_hits]
then normalises them, $h^d_0 = \mathrm{hits}_d / \sum_{d'} \mathrm{hits}_{d'}$ and
$h^d_n = h^d_0 \langle e^{in\psi}\rangle_d$, and sets `total_hits_map`.

## Resolution of the $h$-maps

The $h$-maps vary on very large scales, and can be computed at a lower resolution: the [mapmaking per chunk](mapmaking_chunks.md) takes $h$-maps at a coarser resolution than the output maps and changes their resolution on the fly, each output pixel using the $h$-maps of the coarse pixel that contains it. 