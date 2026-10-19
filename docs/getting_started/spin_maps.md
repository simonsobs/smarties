# Spin Maps

Every map handled by SMARTIES — sky, systematics, $h$-maps, outputs — is stored in a
[`Spin_maps`][smarties.hn.Spin_maps]: an extension of the Python `dict` whose keys are spins and whose values
are NumPy arrays (or pixell `enmap.ndmap` for CAR maps).

## Shapes

| Content | Keys | Value shape |
|---------|------|-------------|
| Sky $S_k$ | `0`, `2`, `-2` | `(npix,)` |
| Systematics $S^{\mathrm{sys},d}_k$ | any spin | `(n_det, npix)` |
| $h$-maps $h^d_n$ | `0, 1, 2, ...` | `(n_det, npix)`; `h[0]` may be `(n_det, 1)` |
| Mapmaker output | `0`, `2`, `-2` | `(npix,)`, holding $I$ and $Q \pm iU$ |

`npix` is the number of **observed** pixels: maps restricted to a mask are flat arrays of
the pixels where the mask is non-zero, for HEALPix and CAR alike. 

Those represent the typical use case of SMARTIES, but the implementation is more general 
and can go beyond CMB applications. 

## Creating Spin Maps

```python
import numpy as np

from smarties import Spin_maps, transform_array_maps_into_spin_maps

rng = np.random.default_rng(0)
I, Q, U = rng.standard_normal((3, 48))

# From a dictionary
sky = Spin_maps.from_dictionary({0: I, 2: (Q + 1j * U) / 2, -2: (Q - 1j * U) / 2})

# From an (nstokes, npix) array of Stokes parameters: same result
sky = transform_array_maps_into_spin_maps(np.array([I, Q, U]))

# From a stack of maps and the list of their spins
outputs = Spin_maps.from_list_maps(np.array([I, (Q - 1j * U) / 2, (Q + 1j * U) / 2]), [0, -2, 2])

print(sky.spins, outputs.spins)
```

`transform_array_maps_into_spin_maps` accepts 1 (I), 2 (Q, U) or 3 (I, Q, U) Stokes
parameters; `n_stokes_output=3` pads the missing ones with zeros.

## Negative Spins

The fields simulated by SMARTIES are real, so $S_{-k} = \overline{S_k}$. A `Spin_maps`
returns the conjugate of the stored positive spin when a negative spin is missing. For
instance, with the $h$-maps of `tests/data` (run from the root of the repository), keeping only
their non-negative spins:

```python
import glob

from smarties import read_lbs_h_maps

files = sorted(path[: -len(".h5")] for path in glob.glob("tests/data/lbs_examples/h_maps_det_*.h5"))
h_maps_negative_positive = read_lbs_h_maps(
    files, prefix="", list_spin=[np.array([n, 0]) for n in range(0, 4)],
    mask=np.ones(12 * 64**2), list_weights=np.ones(len(files)),
    return_only_positive_spins=False # Load both positive and negative spins, Default
)
h_maps = read_lbs_h_maps(
    files, prefix="", list_spin=[np.array([n, 0]) for n in range(0, 4)],
    mask=np.ones(12 * 64**2), list_weights=np.ones(len(files)),
    return_only_positive_spins=True
)

print(h_maps_negative_positive.spins) # -4, -3, -2, -1, 0, 1, 2, 3, 4
print(h_maps.spins) # 0, 1, 2, 3, 4
assert np.allclose(h_maps[-2], np.conj(h_maps[2]))
```

!!! note
    Only item access (`h_maps[-2]`) reconstructs the missing spin. `-2 in h_maps`, `h_maps.keys()`,
    `h_maps.spins` and `h_maps.items()` list the spins actually stored.

## Pixelisation Metadata

`Spin_maps` carries optional metadata, needed by the mapmaking per chunk and set by the
readers of [`smarties.io`][smarties.io]:

| Property | Meaning |
|----------|---------|
| `projection_pixel` | `"healpix"` or `"car"` |
| `shape_fullsky` | full (unmasked) pixel shape, e.g. `(npix,)` or `(ny, nx)` |
| `wcs_car` | WCS of the CAR geometry |
| `total_hits_map` | number of detectors (or hits) per pixel, set when reweighting $h$-maps |

```python
h_maps.projection_pixel = "healpix"
h_maps.shape_fullsky = (12 * 64**2,)
print(h_maps.projection_pixel, h_maps.shape_fullsky, h_maps.wcs_car)  # healpix (49152,) None
```

!!! note
    `Spin_maps.from_dictionary` and arithmetic operations create new objects without
    metadata; set it on the new object when it is needed.

## Operations

```python
total = sky + sky  # new object, union of the keys
sky.add_inplace(sky)  # in place

# Per-detector products, with an einsum signature (default 'd...,d...->d...')
systematics = Spin_maps.from_dictionary({1: rng.standard_normal((3, 48)) + 0j})
scales = {1: np.array([1.0, 1.01, 0.99])}  # e.g. rescale the pointing offset of each detector
systematics.multiply_inplace_detectors_spin_maps(scales, subscripts="d...,d->d...")

# Broadcast detector-independent maps to n_det detectors
common = Spin_maps.from_dictionary({0: I})
common.extend_first_dimension(3)  # common[0].shape == (3, 48)
```

## (n, m) Spins

With a rotating half-wave plate, $h$-maps depend on two spins $(n, m)$ — the crossing angle
and the HWP angle. They are keyed by [`Spin_nm`][smarties.hn.Spin_nm] tuples, with element-wise
negation, addition and subtraction. [`read_lbs_h_maps`][smarties.io.read_lbs_h_maps] returns such
keys; when every $m$ is zero it reduces them to plain integers $n$. The mapmaker currently 
works with integer spins.
