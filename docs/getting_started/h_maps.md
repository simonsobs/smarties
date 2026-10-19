# $h$-maps

The $h$-maps are the only information SMARTIES needs about the scanning strategy. For a detector
$d$ and a pixel $p$,

$$
h^d_n(p) = w^d(p)\, \big\langle e^{in\psi_t} \big\rangle_{t \in p},
\qquad \sum_d w^d(p) = 1,
$$

where the average runs over the samples of detector $d$ falling in pixel $p$, $\psi_t$ is the
crossing angle of the boresight, and $w^d$ the relative weight of the detector (usually its hit
count, normalised over the detectors). They are computed once by a time-ordered data simulator
and reused for any number of sky and systematics realisations; see
[Spin formalism](formalism.md) for their role in the mapmaking.

The snippets of this page read the $h$-maps shipped in `tests/data` of the repository; run them
from the root of the repository.

## Which Spins to Produce

Produce every $h_n$ that couples your systematics to the I, Q, U outputs:

- $h_0, h_2, h_4$ for the sky itself and for spin-0 / spin-2 effects,
- $h_1, h_3$ in addition for pointing offsets (spin 1),
- up to $h_{m_{\max}+2}$ for a beam described up to the azimuthal order $m_{\max}$.

See the table in [Which $h$-maps are needed](formalism.md#which-h-maps-are-needed).

## TOAST / sotodlib $h$-maps

[`read_toast_h_maps`][smarties.io.read_toast_h_maps] reads one file per detector, spin and
component, named

```text
<pair>_<A|B>_<cos|sin>_<n>.<fits|npy|npz|hdf>
```

where `<pair>` identifies a detector pair (here wafer, pixel, telescope and band), `A` / `B` the
two detectors of the pair, and `cos` / `sin` the real and imaginary parts of the $h$-map; the
spin-0 files hold the hit counts. `tests/data/sotodlib_examples` contains such files for 4 pairs
of the Simons Observatory Large Aperture Telescope, on a CAR grid:

```python
import glob

import numpy as np

from smarties import build_mask_from_toast_h_maps, read_toast_h_maps

directory = "tests/data/sotodlib_examples"
pairs = sorted(path[: -len("_A_cos_0.hdf")] for path in glob.glob(f"{directory}/h_n_*_A_cos_0.hdf"))
print(pairs[0])  # tests/data/sotodlib_examples/h_n_w16_p000_LAT_f090

# Observed pixels (at least one detector with a non-zero h_2), detectors per pixel, CAR shape
mask, detector_hits, shape = build_mask_from_toast_h_maps(
    pairs, format_file="hdf", spin_to_read=2, projection_pixel="car",
    return_detectors_hits=True, return_original_shape=True,
)

h_maps = read_toast_h_maps(
    pairs,
    list_spin=[0, 1, 2, 3, 4],  # spin 0: hit counts, used to weight the detectors
    list_prefix=["A_", "B_"],
    mask=mask,
    format_file="hdf",
    projection_pixel="car",
    return_only_positive_spins=True
)
print(sorted(int(spin) for spin in h_maps), h_maps[2].shape)  # spins -4..4, (8 detectors, n_observed)
print(h_maps.projection_pixel, h_maps.shape_fullsky)  # car (345, 1380)
assert np.allclose(h_maps[0].sum(axis=0), 1.0)
```

What the reader does:

- reads `cos` + `i sin` for each requested spin $n \ge 0$ and return for negative spins $h_{-n} = \overline{h_n}$;
- with `mask`, keeps only the pixels where the mask is non-zero, flattened;
- with `reweight_by_hits=True` (default), normalises the maps so that $\sum_d h^d_0 = 1$: from
  the hit counts of the spin-0 files if spin 0 is in `list_spin`, otherwise from the number of
  detectors observing each pixel;
- orders the detectors pair by pair, `A` then `B`;
- records `projection_pixel`, `shape_fullsky` and, for CAR maps, `wcs_car` on the returned
  [`Spin_maps`][smarties.hn.Spin_maps].

!!! tip "Check the normalisation"
    The mapmaker requires $\sum_d h^d_0 = 1$ in every pixel: check
    `h_maps[0].sum(axis=0)` after reading. When spin 0 is not read, pass `list_weights`
    explicitly, one value per detector.

Options worth knowing:

| Argument | Effect |
|----------|--------|
| `format_file` | `"fits"`, `"npy"`, `"npz"`, `"h5"` / `"hdf5"` / `"hdf"` |
| `projection_pixel="car"` | read CAR maps with pixell; maps are flattened, the 2D shape and the WCS are kept as metadata |
| `single_precision=True` | `complex64` maps, halves the memory |
| `duplicate_maps_with_prefix=True` | read `A` only and reuse it for `B` (identical $h$-maps within a pair) |

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
h_maps_lbs = read_lbs_h_maps(
    files,
    prefix="",  # the names above are complete paths
    list_spin=[np.array([n, 0]) for n in range(1, 5)],
    mask=mask_lbs,
    list_weights=np.ones(len(files)),
)
print(sorted(int(spin) for spin in h_maps_lbs), h_maps_lbs[2].shape)  # (8 detectors, 49152 pixels)
```

## Producing $h$-maps

- **TOAST / sotodlib**: the Simons Observatory simulation and mapmaking pipelines can accumulate
  the $h$-maps of each detector while scanning, written in the format above.
- **litebird_sim**: the $h$-maps are produced by the simulation framework of LiteBIRD, in the
  format read by `read_lbs_h_maps`.
- **Planck**: the Planck polmoments are the $h$-maps of the Planck scan; the
  [smarties4planck](https://github.com/camilledemay/smarties4planck) package reads them and
  converts them to the SMARTIES conventions.

Any other simulator can be used: store, for each detector, $\langle e^{in\psi}\rangle$ per
pixel and the hit counts, and build a [`Spin_maps`][smarties.hn.Spin_maps] with
`h_maps[n] = weights * averages[n]` (`weights` normalised over the detectors);
[`reweight_hmaps_by_hits`][smarties.utils.tools.reweight_hmaps_by_hits] applies the
normalisation to un-normalised maps.

## Resolution of the $h$-maps

The $h$-maps vary on larger scales than the sky, and can be computed at a lower resolution:

- the [mapmaking per chunk](mapmaking.md#mapmaking-per-chunk) takes $h$-maps at a larger pixel
  resolution than the output maps and changes their resolution on the fly, each output pixel
  using the $h$-maps of the coarse pixel that contains it;
