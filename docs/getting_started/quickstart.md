# Quickstart

This page runs a first simulation of a systematic effect from end to end, with the $h$-maps
shipped in `tests/data` of the repository. Run the snippets from the root of the repository.
<!-- The [tutorials](../tutorials/litebird_healpix_tutorial.md) go through the same steps in more detail. -->

## The Four Ingredients

A SMARTIES simulation always combines:

1. **$h$-maps** $h^d_n$ — the scanning strategy, one map per detector $d$ and spin $n$, computed
   once through a time-ordered data simulation;
2. **sky spin maps** $S_k$ — the sky signal, common to all detectors;
3. **systematics spin maps** $S^{\mathrm{sys},d}_k$ — the systematic effect, per detector;
4. **the mapmaker**, [`FrameworkSystematics`][smarties.mapmaking.FrameworkSystematics], which
   solves the spin mapmaking equation pixel by pixel.

The [spin formalism](formalism.md) page explains how they fit together.

## Load the $h$-maps

`tests/data/lbs_examples` holds the $h$-maps $h_0, \dots, h_4$ of 8 detectors (4 orthogonal pairs)
of a LiteBIRD-like scan, computed by litebird_sim on HEALPix maps at $N_{\rm side} = 64$. The
last field of each file name is the polarisation angle of the detector, in degrees.

```python
import glob

import healpy as hp
import numpy as np

from smarties import read_lbs_h_maps
from smarties.io import build_mask_from_lbs_h_maps

files = sorted(name[: -len(".h5")] for name in glob.glob("tests/data/lbs_examples/h_maps_det_*.h5"))
polarisation_angles = np.deg2rad([float(name.split("_")[-1]) for name in files])

mask = build_mask_from_lbs_h_maps(files)  # pixels observed by at least one detector
h_maps = read_lbs_h_maps(
    files,
    prefix="",  # the names above are complete paths
    list_spin=[np.array([n, 0]) for n in range(0, 5)],  # (n, m): no half-wave plate, m = 0
    mask=mask,
    list_weights=np.ones(len(files)),
)
print(sorted(int(spin) for spin in h_maps), h_maps[2].shape)  # spins -4..4, (n_det, n_pixels)
assert np.allclose(h_maps[0].sum(axis=0), 1.0)  # the mapmaker requires sum_d h_0 = 1
```
Note that the $h$-maps are loaded with spins $(n,m)$ as per the convention of litbird_sim, with $m = 0$ because there is no half-wave plate. The mapmaker only uses the $n$ index. 
With sotodlib, the current implementation loads $h$-maps with spins $(n)$ only.

## Create the Sky

A CMB sky can be given as spin maps $S_0 = I$ and $S_{\pm 2} = (Q \pm iU)/2$ on the observed pixels,
smoothed by a circular beam of FWHM 2 degrees. The following snippet creates a random realisation of a lensed $\Lambda CDM$ sky with:

```python
from smarties import FrameworkSystematics, transform_array_maps_into_spin_maps

nside, lmax = hp.npix2nside(mask.size), 128
framework = FrameworkSystematics(map_shape=(mask.size,), nstokes=3, lmax=lmax)
sky = framework.get_CMB_spin_sky_maps(nside=nside, fwhm=np.deg2rad(2.0), seed=0)
```

With CAMB installed, a lensed ΛCDM realisation is one call away:
[`create_CMB_spin_maps(nside, nstokes=3, lmax=lmax, fwhm=120.0)`][smarties.sky.cmb.create_CMB_spin_maps]. 
The `sky` variable is a dictionary with keys `0`, `2`, `-2` holding the spin maps $S_0$, $S_2$ and $S_{-2}$.

## Mapmaking Without Systematics

Without systematics, the mapmaker returns the input sky in every pixel where the mapmaking
matrix can be inverted (so for pixels with a sufficiently good cross-linking). 
Its inverse only depends on the $h$-maps and the polarisation angles, and can be kept for later runs:

```python
maps, inverse_matrix = framework.compute_total_maps(
    h_maps, sky, mask=mask, polar_angle=polarisation_angles,
    return_Q_U=True, return_inverse_mapmaking_matrix=True,
)
initial_iqu = transform_spin_maps_into_array_maps(sky)
print("max |output - input|:", np.abs(maps.real - initial_iqu[:, mask != 0]).max())
print("condition number:", np.linalg.cond(inverse_matrix))
```

## Add a Systematic Effect

A differential pointing offset of each detector leaks temperature into polarisation:

```python
from smarties import create_pointing_spin_leakage_map_BICEP

rng = np.random.default_rng(1)
delta_x, delta_y = np.deg2rad(rng.normal(0, 1 / 60, (2, len(files))))  # offsets in radians
leakage = create_pointing_spin_leakage_map_BICEP(sky[0], delta_x, delta_y, lmax=lmax, mask=mask)
print({int(spin): value.shape for spin, value in leakage.items()})  # spins -1, +1: (n_det, n_pixels)

maps = framework.compute_total_maps(
    h_maps, sky, leakage, mask=mask, polar_angle=polarisation_angles,
    inverse_mapmaking_matrix=inverse_matrix, return_Q_U=True,
)
residual = maps.real - initial_iqu[:, mask != 0]
print("rms residual (I, Q, U):", residual.std(axis=1))
```

`residual` is the systematic map: what the pointing offsets add to the sky once the data are
binned with this scan. Without `return_Q_U=True` the result is a
[`Spin_maps`][smarties.hn.Spin_maps] with keys `0`, `2`, `-2` holding $I$ and $Q \pm iU$.

## Package Layout

| Module | Content |
|--------|---------|
| [`smarties.hn`][smarties.hn] | [`Spin_maps`][smarties.hn.Spin_maps], the container for spin maps and $h$-maps, and [`Spin_nm`][smarties.hn.Spin_nm] for $(n, m)$ spins |
| [`smarties.mapmaking`][smarties.mapmaking] | [`FrameworkSystematics`][smarties.mapmaking.FrameworkSystematics], the spin mapmaker |
| [`smarties.io`][smarties.io] | readers for TOAST / sotodlib and litebird_sim $h$-maps, masks, YAML parameters; writers for output maps |
| [`smarties.systematics`][smarties.systematics] | pointing, beamwidth and ellipticity spin maps and templates |
| [`smarties.sky`][smarties.sky] | CMB generation, point sources, harmonic-space beam convolution |
| [`smarties.instrument`][smarties.instrument] | detector lists and parameters |
| [`smarties.external.s4cmb`][smarties.external.s4cmb] | spin-raising / lowering operators and spherical derivatives |
| [`smarties.utils`][smarties.utils] | harmonic transforms, chunking and resolution tools, interpolation, MASTER spectra |

Most public functions are re-exported at the top level, so `from smarties import ...` works for
all of them.

## Next Steps

- [Spin formalism](formalism.md): the equations behind the mapmaker.
- [$h$-maps](h_maps.md): reading the outputs of TOAST / sotodlib and litebird_sim.
- [Systematics](systematics.md): the available systematic effects and their conventions.
- [Mapmaking](mapmaking.md): masks, polarisation angles, mapmaking per chunk.
<!-- - [Tutorials](../tutorials/litebird_healpix_tutorial.md): complete runs on HEALPix and CAR maps. -->
