# Spin Formalism

SMARTIES never builds a timestream. It relies on the fact that, for a **binned mapmaker**,
the output maps depend on the scan only through a few pixel-domain averages, the
**$h$-maps**. This page derives the equation solved by
[`FrameworkSystematics.compute_total_maps`][smarties.mapmaking.FrameworkSystematics.compute_total_maps]
and fixes the conventions used everywhere in the package.

## Spin Decomposition of the Data

A detector $d$ with polarisation angle $\gamma^d$ observes pixel $p$ at sample $t$ with the
boresight (crossing) angle $\psi_t$. Without systematics its data read

$$
d_t = I + Q \cos 2(\psi_t + \gamma^d) + U \sin 2(\psi_t + \gamma^d).
$$

Writing the sky as **spin maps**

$$
S_0 = I, \qquad S_{\pm 2} = \frac{Q \pm iU}{2},
$$

the same timestream becomes a sum of spin-$k$ harmonics of the total angle,

$$
d_t = \sum_{k \in \{0, \pm 2\}} S_k \, e^{-ik(\psi_t + \gamma^d)} .
$$

A systematic effect adds its own spin-$k$ terms, which may differ from detector to detector
and are defined with respect to the boresight angle:

$$
d_t = \sum_{k} S_k \, e^{-ik(\psi_t + \gamma^d)} \;+\; \sum_{k} S^{\mathrm{sys},d}_k \, e^{-ik\psi_t}.
$$

Because the timestream is real, $S_{-k} = \overline{S_k}$ and
$S^{\mathrm{sys},d}_{-k} = \overline{S^{\mathrm{sys},d}_k}$. Typical spins are $k = \pm 1$
for pointing offsets, $k = 0$ for beamwidth mismatch, $k = \pm 2$ for beam
ellipticity; an asymmetric beam populates all $k \le m_{\max}$ of the beam.

## $h$-maps

The scan enters through the **$h$-maps**, which are typically built as $h^d_{n,\rm TOD}(p) = \, \big\langle e^{in\psi_t} \big\rangle_{t \in p}$. `SMARTIES` will read them from disk and re-define them as

$$
h^d_n(p) = w^d(p) \, \big\langle e^{in\psi_t} \big\rangle_{t \in p},
\qquad \sum_d w^d(p) = 1 ,
$$

where the average runs over the samples of detector $d$ falling in pixel $p$, and the
weights $w^d$ (usually hit counts, possibly inverse noise variances) are normalised over the
detectors. In particular $h^d_0 = w^d$, and $h^d_{-n} = \overline{h^d_n}$.

!!! info "Normalisation"
    [`compute_total_maps`][smarties.mapmaking.FrameworkSystematics.compute_total_maps]
    checks that $\sum_d h^d_0 = 1$ in every pixel and raises an `AssertionError` otherwise.
    The readers in [`smarties.io`][smarties.io] normalise the maps for you; see
    [$h$-maps](h_maps.md).

## The Mapmaking Equation

A binned mapmaker solves, pixel by pixel, the normal equations of the model
$d_t = I + Q\cos 2\alpha_t + U \sin 2\alpha_t$ with $\alpha_t = \psi_t + \gamma^d$. Projecting
them on the spin basis gives a small linear system for each pixel,

$$
\sum_{k'} M_{kk'} \, x_{k'} = \tilde d_k , \qquad k, k' \in \{0, \pm 2\},
$$

with the **data vector**

$$
\tilde d_k = f_k \sum_d e^{ik\gamma^d} \sum_{k'} h^d_{k-k'}
\Big( S^{\mathrm{sys},d}_{k'} + e^{-ik'\gamma^d}\, S_{k'} \Big),
$$

the **mapmaking matrix**

$$
M_{kk'} = f_k f_{k'} \sum_d e^{i(k-k')\gamma^d} \, h^d_{k-k'} ,
$$

and $f_0 = 1$, $f_{\pm 2} = 1/2$. The solution is

$$
x_0 = \hat{I}, \qquad x_{\pm 2} = \hat{Q} \pm i \hat{U} .
$$

This can be reformulated as

$$
\begin{pmatrix}
\hat{I} \\
\hat{Q} + i \hat{U} \\
\hat{Q} - i\hat{U}
\end{pmatrix}
=
\begin{pmatrix}
h^{tot}_{0} & \frac{1}{2} h^{tot}_2 & \frac{1}{2} h^{tot}_{-2} \\
\frac{1}{2} h^{tot}_{-2} & \frac{1}{4} & \frac{1}{4} h^{tot}_{-4} \\
\frac{1}{2} h^{tot}_2 & \frac{1}{4} h^{tot}_{4} & \frac{1}{4} 
\end{pmatrix}
^{-1}
\begin{pmatrix}
\sum_d \ \ \quad \ \  w^d \sum_{n \in \mathbb{Z}}  h_{0- n}^d \mathrm{S}_{n}^d \\
\frac{1}{2}\sum_d  e^{i2\gamma^d} \  w^d  \sum_{n \in \mathbb{Z}}  h_{2- n}^d \mathrm{S}_{n}^d \\
\frac{1}{2}\sum_d \, e^{-i2\gamma^d} w^d\,  \sum_{n \in \mathbb{Z}}  h_{-2- n}^d \mathrm{S}_{n}^d
\end{pmatrix}
$$

with $h^{tot}_k = \sum_d e^{i k \gamma^d} w^d \sum_{n \in \mathbb{Z}} h_{k- n}^d$. 
The normalization $w^d (p)$ is a function of the detector $d$ and pixel $p$, and can be typically defined as (without accounting for efficiency differences between detectors):

$$
w^d(p) = \frac{N_{\mathrm hits}^d (p)}{N_{\mathrm hits}^{\mathrm tot} (p)}
$$ 

with $N_{\mathrm hits}^{y} (p)$ the number of hits in a pixel $p$ either for a given detector d, represented here by $y = d$, or for all detectors crossing this pixel as represented by $y = \rm tot$. 
The sums over $n$ run over the spins of the sky and systematics present in the simulation.

In the well-conditioned pixels and without systematics $x$ is exactly the input sky : the systematics maps
are the only source of residuals. 

`FrameworkSystematics` builds $M$ from the $h$-maps and polarisation angles $\gamma^d$
([`get_inverse_mapmaking_matrix`][smarties.mapmaking.FrameworkSystematics.get_inverse_mapmaking_matrix]),
the data vector from the $h$-maps, polarisation angles, sky and systematics
([`compute_spin_convolution`][smarties.mapmaking.FrameworkSystematics.compute_spin_convolution]),
and applies $M^{-1}$. Since $M$ only depends on the scan, its inverse can be computed once
and reused for many systematics realisations.

!!! success "Exact for a binned mapmaker"
    The equation above is not an approximation with respect to the usual time domain approaches: simulating the timestreams sample by sample and binning them with an $(I, Q, U)$ pointing matrix gives the same maps to machine precision. 

## Condition Number

The mapmaking matrix can only be inverted where the pixels are well cross-linked, meaning that the detector orientations are diverse enough
to separate $I$, $Q$ and $U$. Its condition number measures it, pixel by pixel: it equals 4
for a perfect angle coverage with $I, Q, U$ (1 for $Q, U$ only), because of the factors $f_k$,
and grows when the coverage degrades. In pixels observed with too few orientations (typically
the edges of a scan), neither timestream nor pixel-based methods can recover the sky. 
One may build a mask by selecting the well-conditioned pixels from the inverse 
matrix returned by the mapmaker:

```python
import numpy as np

condition_number = np.linalg.cond(inverse_mapmaking_matrix)  # (npix,)
well_conditioned = np.isfinite(condition_number) & (condition_number < 9)
```

Without systematics, the output equals the input sky to numerical precision in these pixels.

## Which $h$-maps Are Needed

A signal of spin $k'$ reaches the output spin $k$ through $h_{k-k'}$. With outputs
$k \in \{0, \pm 2\}$:

| Signal | Spins $k'$ | $h$-maps used $\lvert n \rvert$ |
|--------|-----------|-------------------------------|
| Sky (I, Q, U) | $0, \pm 2$ | 0, 2, 4 |
| beamwidth mismatch | $0$ | 0, 2 |
| Pointing offset | $\pm 1$ | 1, 3 |
| Beam ellipticity | $\pm 2$ | 0, 2, 4 |
| Spin-3 beam asymmetry | $\pm 3$ | 1, 3, 5 |
| Spin-4 beam asymmetry | $\pm 4$ | 2, 4, 6 |

!!! tip "Load every $h$-map your systematics need"
    The coupling sum runs over the $h$-maps present in the dictionary. With only
    $h_0, h_2, h_4$ loaded, a pointing systematic has no effect on the output maps: load
    every $h_n$ of the table above for the systematics you simulate.

## Polarisation Angles and Efficiencies

- `polar_angle` ($\gamma^d$, radians, one per detector) rotates the sky relative to the
  boresight. Systematics maps stay attached to the boresight frame, as in the equations
  above.
- `polar_efficiency_coeff` ($\varepsilon^d$) multiplies the spin $\pm 2$ rows and columns of
  the system, i.e. it is the efficiency *assumed by the mapmaker*; the simulated data are not
  rescaled. For a single efficiency $\varepsilon$ the output $Q, U$ are the input divided by
  $\varepsilon$.

## Real Fields and Positive Spins

All the fields above are real, so the negative-spin maps are the complex conjugates of the
positive ones. [`Spin_maps`][smarties.hn.Spin_maps] uses this: a negative spin that is not
stored is returned as the conjugate of the positive one, and the mapmaker
(`optimize_computations=True`) only computes the non-negative spins of the data vector and
of the solution. Storing only $n \ge 0$ halves the memory taken by the $h$-maps and the
systematics maps.

## Further Reading

- McCallum, Thomas, Brown & Tessore, *Spin characterization of systematics in CMB surveys —
  a comprehensive formalism*, MNRAS 501, 802 (2021),
  [doi:10.1093/mnras/staa3609](https://doi.org/10.1093/mnras/staa3609).
- McCallum, Thomas & Brown, *Fast map-based simulations of systematics in CMB surveys
  including effects of the scanning strategy*, MNRAS 513, 3610 (2022),
  [doi:10.1093/mnras/stac561](https://doi.org/10.1093/mnras/stac561).
- BICEP2 Collaboration, *BICEP2 III: Instrumental systematics*, ApJ 814, 110 (2015),
  [arXiv:1502.00608](https://arxiv.org/abs/1502.00608) — the beam templates.
- Demay, C. et al., *Exact map-space beam convolution via spin-mapmaking*,
  [arXiv:2610.20231](https://arxiv.org/abs/2610.20231) (2026).
