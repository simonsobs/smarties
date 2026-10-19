# Systematics

A systematic effect is described by its spin maps $S^{\mathrm{sys},d}_k$, one
`(n_det, npix)` array per spin, defined in the boresight frame (see
[Spin formalism](formalism.md)). Once these maps are built, the mapmaker propagates the effect
to the I, Q, U maps without any operation on time-ordered data. SMARTIES provides two families
of generators:

- **Taylor-expansion templates** (BICEP2 III, [arXiv:1502.00608](https://arxiv.org/abs/1502.00608)):
  differential pointing, beamwidth and ellipticity, computed from derivatives of the
  temperature map. Cheap, and linear in the per-detector parameters.
- **Exact beam convolution** from the beam harmonic coefficients $b_{\ell m}$, for arbitrary
  beam shapes ([Demay et al. 2026](https://arxiv.org/abs/2610.20231)).

All generators return a [`Spin_maps`][smarties.hn.Spin_maps] that can be passed directly as
`spin_systematics_maps` to the mapmaker.

## Conventions

- Detector coordinates: $x$ points along $-\hat\theta$ and $y$ along $-\hat\phi$ when the
  detector is at rest; both rotate with the boresight angle.
- Angles are in **radians**, beamwidths (`sigma_fwhm`, `fwhm`) are **FWHM in arcmin**.
- The input temperature map is the sky **already convolved with the nominal circular
  beam**; the templates describe the difference with respect to that beam.
- Spin derivatives follow the standard convention for a spin-0 field,
  $\eth f = -(\partial_\theta + \tfrac{i}{\sin\theta}\partial_\phi) f$ and
  $\bar\eth f = -(\partial_\theta - \tfrac{i}{\sin\theta}\partial_\phi) f$.

The examples on this page use the following sky:

```python
import healpy as hp
import numpy as np

nside, lmax = 32, 64
ell = np.arange(lmax + 1)
cl = np.zeros(lmax + 1)
cl[2:] = 1.0 / ell[2:] ** 2
np.random.seed(3)
temperature = hp.smoothing(hp.synfast(cl, nside, lmax=lmax), fwhm=np.deg2rad(1.0))
n_det = 2
```

## Differential Pointing

A detector pointing at $\hat n + \hat{\delta}$ instead of $\hat n$ sees, at first
order, a spin $\pm 1$ signal

$$
S^{\mathrm{sys},d}_{-1} = \frac{\delta x_d - i\,\delta y_d}{2}\; i\bar\eth I,
\qquad S^{\mathrm{sys},d}_{+1} = \overline{S^{\mathrm{sys},d}_{-1}} .
$$

```python
from smarties import create_pointing_spin_leakage_map, create_pointing_spin_leakage_map_BICEP

delta_x = np.deg2rad([1.0, -0.5]) / 60  # 1 and -0.5 arcmin
delta_y = np.deg2rad([0.0, 0.8]) / 60
pointing = create_pointing_spin_leakage_map_BICEP(temperature, delta_x, delta_y, lmax=lmax)
print(sorted(pointing.keys()), pointing[1].shape)  # [-1, 1] (n_det, npix)

# Same effect in amplitude / angle form: delta_x - i delta_y = -rho exp(i chi)
offsets = -(delta_x - 1j * delta_y)
polar = create_pointing_spin_leakage_map(temperature, np.abs(offsets), np.angle(offsets), lmax=lmax)
assert np.allclose(polar[-1], pointing[-1])
```

Spin-1 systematics only reach the maps through $h_1$ and $h_3$: load them.

## Differential Beamwidth

A beamwidth $\sigma + \delta\sigma_d$ instead of $\sigma$ gives a spin-0 signal

$$
S^{\mathrm{sys},d}_0 = \sigma\,\delta\sigma_d \left(\partial^2_\theta I + \frac{1}{\sin^2\theta}\partial^2_\phi I\right),
$$

with $\sigma$ the Gaussian width of the nominal beam (from `sigma_fwhm`) and
$\delta\sigma_d$ in radians.

```python
from smarties import create_dict_template_beamwidth_leakage_map_BICEP

parameters, templates = create_dict_template_beamwidth_leakage_map_BICEP(
    temperature, delta_sigma=np.array([1e-5, -2e-5]), sigma_fwhm=60.0, lmax=lmax
)
beamwidth = {0: np.outer(parameters[0], templates[0])}  # (n_det, npix)
```

## Differential Ellipticity

An elliptical Gaussian beam leaks temperature into polarisation through a spin $\pm 2$
signal

$$
S^{\mathrm{sys},d}_{-2} = \frac{\sigma^2}{2}\,(p_d - i c_d)\,
\big[\eth\eth I - \cot\theta\;\eth I\big],
\qquad S^{\mathrm{sys},d}_{+2} = \overline{S^{\mathrm{sys},d}_{-2}},
$$

where $(p, c)$ are the plus / cross ellipticities. The ellipticity can be given in any of the
conventions of
[`convert_ellipticities_conventions`][smarties.utils.tools.convert_ellipticities_conventions]:

| `ellipticity_parameter_convention` | Parameter | Circular beam |
|------------------------------------|-----------|---------------|
| `"Third flattening"` | $f = (a - b)/(a + b)$, with angle | $0$ |
| `"Third eccentricity"` | $(a^2 - b^2)/(a^2 + b^2)$, with angle | $0$ |
| `"Modified second flattening"` | $a/b$, with angle | $1$ |
| `"Plus-Cross ellipticity"` | $p = 2f\cos 2\varphi$, $c = 2f\sin 2\varphi$ (keys `dp`, `dc`) | $p = c = 0$ |

```python
from smarties import get_differential_ellipticity_BICEP

ellipticity = {
    "ellipticity_value": np.array([0.02, 0.05]),  # third flattening
    "ellipticity_angle": np.array([0.3, 1.2]),  # radians
    "ellipticity_parameter_convention": "Third flattening",
}
elliptical = get_differential_ellipticity_BICEP(temperature, ellipticity, sigma_fwhm=60.0, lmax=lmax)
print(sorted(elliptical.keys()))  # [-2, 0, 2], spin 0 is zero
```

[`get_differential_ellipticity_no_calibration`][smarties.systematics.ellipticity.get_differential_ellipticity_no_calibration]
implements the alternative second-order expansion of
[arXiv:2011.13910](https://arxiv.org/abs/2011.13910), which also has a spin-0 term.

## Templates

Each Taylor systematic factorises into a per-detector parameter times a sky template that
does not depend on the detector. The `create_dict_template_*` functions return the two
factors separately:

```python
from smarties import create_dict_template_pointing_BICEP

parameters, templates = create_dict_template_pointing_BICEP(temperature, delta_x, delta_y, lmax=lmax)
rebuilt = np.outer(parameters[-1], templates[-1])
assert np.allclose(rebuilt, pointing[-1])
```

Storing the templates instead of the `(n_det, npix)` maps saves memory for large focal
planes; the mapmaking per chunk builds the maps on the fly from them
(`build_systematics_maps_from_templates=True`, see [Mapmaking](mapmaking.md)).

## Spherical Derivatives

The templates need first and second derivatives of the temperature map. They are computed
from its harmonic coefficients by default, using the spin-raising and lowering operators of
[`smarties.external.s4cmb`][smarties.external.s4cmb]. To compute them once for many runs, or
for CAR maps, pass them through `spherical_derivatives`:

| Key | Content |
|-----|---------|
| `"theta"`, `"phi"` | $\partial_\theta I$, $\frac{1}{\sin\theta}\partial_\phi I$ |
| `"theta_theta"`, `"phi_phi"` | $\partial^2_\theta I$, $\frac{1}{\sin^2\theta}\partial^2_\phi I$ |
| `"theta_phi"` | $\frac{1}{\sin\theta}\partial_\theta\partial_\phi I$ |

[`read_spherical_derivatives_from_file`][smarties.io.read_spherical_derivatives_from_file]
reads them from a FITS file with six maps: the map itself, then the five derivatives in the
order of the table.

For CAR maps, provide the derivatives through `spherical_derivatives`.

## Exact Beam Convolution

For an arbitrary beam, [`get_beam_convolution_spins_maps`][smarties.sky.convolution.get_beam_convolution_spins_maps]
computes the spin-$k$ maps of each detector exactly, from the sky $a_{\ell m}$ (T, E, B) and
the beam $b_{\ell m}$ up to $m_{\max}$. Passed to the mapmaker with the $h$-maps of a scan,
they reproduce the maps of a full $4\pi$ beam convolution followed by a binned mapmaking, without
time-ordered data ([Demay et al. 2026](https://arxiv.org/abs/2610.20231)):

```python
from smarties import gaussian_circular_beam_blms, get_beam_convolution_spins_maps

sky_alms = hp.map2alm(hp.synfast([cl, cl / 10, cl / 100, 0 * cl], nside, lmax=lmax, new=True), lmax=lmax)
beam_blms = gaussian_circular_beam_blms(np.deg2rad(1.0), lmax, mmax=4, pol_angle_rad=0.0)

beam_maps = get_beam_convolution_spins_maps(
    alms={"det0": sky_alms},
    blms={"det0": beam_blms},
    det_names=["det0"],
    lmax=lmax,
    mmax_beam=4,
    shape_pixels_output=(hp.nside2npix(nside),),
)
print(sorted(int(spin) for spin in beam_maps))  # [0, 1, 2, 3, 4]: negative spins by conjugation
```

The result contains the whole beam-convolved signal, sky included. Use it as
`spin_systematics_maps` with zero sky maps, or set `substract_gaussian_beam=True` (with
`fwhm` and `pol_angles_rad`) to keep only the difference with a circular Gaussian beam and
add the Gaussian-smoothed sky as `spin_sky_maps`. Pass `wcs` to compute CAR maps. 
<!-- The [LiteBIRD tutorial](../tutorials/litebird_healpix_tutorial.md) runs the whole chain, from
$b_{\ell m}$ to the output maps. -->

!!! example "Planck beams: smarties4planck"
    [smarties4planck](https://github.com/camilledemay/smarties4planck) is a self-consistent
    package interfacing SMARTIES with the Planck beam products: it reads the Planck beam
    $b_{\ell m}$ and polmoments ($h$-maps), converts them to the SMARTIES conventions and
    computes beam-convolved Planck maps from arbitrary input skies. It is a complete example of
    the beam-convolution workflow with real instrument data.
