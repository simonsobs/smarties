# SMARTIES Documentation

**SMARTIES** (*Spin MAp Reduction Toolkit for systematics Impact EStimation*) simulates the
impact of instrumental systematics **directly at the map level** accurately with respect to
time-ordered data (TOD) computations, without the need for full-scale simulations. It extends the spin-based formalism of
[McCallum et al. (2021, 2022)](https://doi.org/10.1093/mnras/stac561) and is developed within the Simons Observatory and LiteBIRD.

The scanning strategy information enters through the so-called **$h$-maps**, map reduction of the pointing information built as pixel averages of
$e^{in\psi}$ over the detector parallactic angles $\psi$ (also called detector orientations). 
Both sky maps and systematic effects that are described as stationary (without intrinsic time variation) will be ingested as **spin maps**. 
Combining the two with the spin mapmaking equation gives estimates of the I, Q, U maps, which are equivalent to what a binned mapmaker would have produced, at a fraction of the cost. 
The numerical cost scales with the number of pixels observed, rather than with the number of time samples. 

## Key Features

🚫 **Propagating systematics without TOD.** Instrumental systematics are propagated to the maps without any operation on time-ordered data: once the $h$-maps of a scan are computed, any new systematic effect, a sky or detector parameters only cost a mapmaking at the map level, scaling in the number of pixels observed.

🗺️ **HEALPix and CAR.** SMARTIES works on HEALPix maps and on CAR maps (plate carrée projection) through
[pixell](https://github.com/simonsobs/pixell).

🔭 **Exact $4\pi$ beam convolution from $b_{\ell m}$.** The harmonic coefficients $b_{\ell m}$ of each detector beam can be ingested to replicate a full $4\pi$ beam convolution, again without
TOD operation ([Demay et al. 2026](https://arxiv.org/abs/2610.20231)). Template-based instrumental systematics à la BICEP replicating elliptical beam, beam pointing offset and beamwidth mismatch are also available.

🔍 **On-the-fly adaptive resolution.** The $h$-maps can be provided at a lower resolution than the output maps: routines are implemented to changes their resolution on the fly during the mapmaking stage, without needing to store them at the output resolution.

📡 **Detector-wise interpolation.** On a given focal plane, one can provide $h$-maps coming from a subset of detectors, and these can be used to interpolate the missing ones over the full focal plane on-the-fly during the mapmaking stage.

🛰️**Simulation interfaces.** Interfaces are built to ingest the $h$-maps produced by TOAST / sotodlib and litebird_sim. 

## Getting Started

```bash
git clone https://github.com/simonsobs/smarties.git && cd smarties
pip install .
```

SMARTIES requires Python ≥ 3.13. For the best performance of the spherical-harmonic transforms,
[ducc0](https://gitlab.mpcdf.mpg.de/mtr/ducc) should be compiled from source on your machine:
see [Installation](getting_started/installation.md), then the
[Quickstart](getting_started/quickstart.md) for a first run.

## Quick Example

The repository ships, in `tests/data` (to be retrieved with `git lfs pull`), the $h$-maps of 8 detectors of a LiteBIRD-like scan. From
the root of the repository:

```python
import glob

import healpy as hp
import numpy as np

from smarties import (
    FrameworkSystematics,
    create_pointing_spin_leakage_map_BICEP,
    read_lbs_h_maps,
    transform_array_maps_into_spin_maps,
    transform_spin_maps_into_array_maps
)
from smarties.io import build_mask_from_lbs_h_maps

# h-maps: one file per detector, normalised so that sum_d h_0 = 1 in every pixel
files = sorted(name[: -len(".h5")] for name in glob.glob("tests/data/lbs_examples/h_maps_det_*.h5"))
angles = np.deg2rad([float(name.split("_")[-1]) for name in files])  # polarisation angles
mask = build_mask_from_lbs_h_maps(files)
h_maps = read_lbs_h_maps(
    files, prefix="", list_spin=[np.array([n, 0]) for n in range(0, 5)],
    mask=mask, list_weights=np.ones(len(files)),
)

# Sky: I, Q, U -> spin maps S_0 = I, S_{+-2} = (Q +- iU) / 2
nside, lmax = hp.npix2nside(mask.size), 128
framework = FrameworkSystematics(map_shape=(mask.size,), nstokes=3, lmax=lmax)
sky = framework.get_CMB_spin_sky_maps(nside=nside, fwhm=np.deg2rad(2.0), seed=0)

# Alternatively, one can provide a custom sky in I, Q, U format and convert it to spin maps:
# iqu = hp.read_map('path/to/iqu_map.fits', field=(0, 1, 2))
# sky = transform_array_maps_into_spin_maps(iqu)

# Systematic: differential pointing offsets (radians) of each detector
rng = np.random.default_rng(1)
offsets = np.deg2rad(rng.normal(0, 1 / 60, (2, len(files))))
leakage = create_pointing_spin_leakage_map_BICEP(sky[0], offsets[0], offsets[1], lmax=lmax)

# Mapmaking: binned I, Q, U maps including the systematic
maps = framework.compute_total_maps(h_maps, sky, leakage, polar_angle=angles, return_Q_U=False)
initial_iqu = transform_spin_maps_into_array_maps(sky)
final_iqu = transform_spin_maps_into_array_maps(maps)
print("rms of the T -> P leakage in Q, U:", (initial_iqu.real - final_iqu)[1:].std(axis=1))
```

## Documentation Contents

- **Getting Started**: [Installation](getting_started/installation.md),
  [Quickstart](getting_started/quickstart.md), [Spin formalism](getting_started/formalism.md),
  [Spin maps](getting_started/spin_maps.md), [$h$-maps](getting_started/h_maps.md),
  [Systematics](getting_started/systematics.md), [Mapmaking](getting_started/mapmaking.md)
<!-- - **Tutorials**: [LiteBIRD-like scan on HEALPix maps](tutorials/litebird_healpix_tutorial.md),
  [Simons Observatory scan on CAR maps](tutorials/sotodlib_car_tutorial.md) -->
- **API Reference**: generated from the [`smarties`](api/smarties/index.md) source tree, one page
  per module

## Packages Built on SMARTIES

[**smarties4planck**](https://github.com/camilledemay/smarties4planck) is a self-consistent
package interfacing SMARTIES with the Planck beam products: it reads the Planck beam harmonic
coefficients $b_{\ell m}$ and polmoments (the Planck $h$-maps), converts them to the SMARTIES
conventions and simulates beam-convolved Planck maps from arbitrary input skies.

## Citing SMARTIES

If you use SMARTIES, please contact `magdy.morshed at fe.infn.it` and consider citing:

> The Simons Observatory Collaboration, *The Simons Observatory: A framework for efficient
> map-based simulations of instrumental systematics for CMB surveys* (in prep.).

> Demay, Morshed, Gerbino & Pagano, *Exact map-space beam convolution via spin-mapmaking*,
> [arXiv:2610.20231](https://arxiv.org/abs/2610.20231) (2026).

The formalism builds on McCallum, Thomas & Brown, *Fast map-based simulations of systematics in
CMB surveys including effects of the scanning strategy*, MNRAS 513, 3610 (2022),
[doi:10.1093/mnras/stac561](https://doi.org/10.1093/mnras/stac561), and on McCallum et al.,
*Spin characterization of systematics in CMB surveys — a comprehensive formalism*, MNRAS 501,
802 (2021), [doi:10.1093/mnras/staa3609](https://doi.org/10.1093/mnras/staa3609).
