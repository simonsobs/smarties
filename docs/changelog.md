# Changelog

All notable changes to SMARTIES will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project will adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once
releases are tagged.

## [Unreleased]

### Added

- `FrameworkSystematicsChunks`: mapmaking per chunk of pixels on HEALPix and CAR maps, with $h$-maps at a coarser resolution than the output maps, $h$-maps interpolated on the focal plane to detectors without simulated $h$-maps (wafer by wafer, from the hits and averages of the known detectors), and systematics built from templates chunk by chunk; see [Mapmaking per chunk](getting_started/mapmaking_chunks.md).
- Adaptive-resolution tools for HEALPix and CAR maps in `smarties.utils.resolution`:
  `get_ratio_resolution`, `get_parent_pixels`, `get_children_pixels_healpix`,
  `downgrade_mask`, `get_chunks_healpix`, `get_pixels_chunk`, `get_projector_chunk`,
  `upgrade_h_maps`.
- Numba kernels for the spin convolution (about ten times faster) of the mapmaking matrices. 
- `read_toast_h_maps` and `read_lbs_h_maps` return the attributes of the files (polarisation angles, focal-plane positions, ...), `return_attributes=True` by default. 
- `Spin_maps.total_hits_map`, set by the $h$-map readers, and `Spin_maps.copy_sliced`.
- Checks that the $h$-maps needed by the spins of the sky and systematics maps are given.
- Warning when the file of spherical derivatives is not found (`read_spherical_derivatives_from_file` returns `None`).
- Documentation pages [Mapmaking per chunk](getting_started/mapmaking_chunks.md) and [Logs](getting_started/logs.md); this changelog in the navigation.
- Documentation site (Zensical + mkdocstrings) with a getting-started guide and an API reference generated from the docstrings; Read the Docs configuration.
- `Spin_maps` returns the conjugate of the stored spin when a negative spin is missing, so that only non-negative spins need to be stored.
- Optimised mapmaking (`optimize_computations=True`): only the non-negative spins of the data vector and of the solution are computed.
- `single_precision` option for the mapmaker, the $h$-map readers and the beam convolution.
- Numba implementation of `almxfl`, used by the beam convolution.
- `get_beam_convolution_spins_maps(return_only_positive_spins=True)`.
- Logging through the `logging` module (`smarties` logger, level from `smarties_LOGLEVEL`) instead of `print`.
- `uv.lock`, pinning the dependencies for [uv](https://docs.astral.sh/uv/) environments (`uv sync`).

### Changed

- **Breaking:** `FrameworkSystematics(list_spin_output=...)` defaults to `[0, -2, 2]` and must be symmetric around 0; the output no longer depends on the ordering of the spins. The layout of the inverse mapmaking matrix changed accordingly: recompute saved inverse matrices.
- **Breaking:** `save_extended_final_maps` is renamed `save_spin_maps`.
- `compute_total_maps` and its optimised variant are merged; docstrings updated across the package.
- **Breaking:** `read_toast_h_maps` and `build_mask_from_toast_h_maps` read the current sotodlib format: one HDF5 file per detector, with groups `"n,0"` (`Re`, `Im`) and the geometry in the attributes. `list_prefix`, `projection_pixel` and `duplicate_maps_with_prefix` (reader), `list_prefix`, `spin_to_read`, `projection_pixel` and `single_precision` (mask) are removed; `format_file` defaults to `"hdf"` (was `"npz"`), and `list_spin` of the reader to `[0, 1, 2, 3, 4]` (was `[0, 2, 4]`).
- **Breaking:** `read_toast_h_maps` and `read_lbs_h_maps` return `(h_maps, attributes)` by default (`return_attributes=False` for the $h$-maps only).
- **Breaking:** `read_lbs_h_maps` needs the spin `(0, 0)` in `list_spin` with `reweight_by_hits=True` (default).
- **Breaking:** `reweight_hmaps_by_hits(h_dictionary, in_place=False, spin_0=0)` takes the hits from the spin-0 maps (`list_weights`, `list_spin` and `error_precision` removed) and sets `total_hits_map`.
- **Breaking:** `compute_total_maps(mask_input=...)` is removed: the inputs are given on the observed pixels.
- **Breaking:** the log level is read from `smarties_LOGLEVEL` (was `LOGLEVEL`). The `smarties` logger prints through its own handler, and no longer passes its messages to the root logger when the latter has no handler at import (no duplicated messages); see [Logs](getting_started/logs.md).
- `get_row_mapmaking_matrix` uses less memory.
- The litebird_sim $h$-maps of `tests/data/lbs_examples` are updated.

### Deprecated

- `FrameworkSystematics.compute_total_maps_chunks`: use `FrameworkSystematicsChunks.compute_total_maps_chunks` ([migration](getting_started/mapmaking_chunks.md#migrating-from-frameworksystematicscompute_total_maps_chunks)).
- `read_toast_h_maps_old` and `build_mask_from_toast_h_maps_old`, readers of the previous sotodlib format (one file per detector, spin and component).

### Fixed

- `FrameworkSystematics.compute_total_maps_chunks` accepts the $h$-maps of the positive spins only (it stopped on `KeyError: -2`), and includes the couplings through the $h$-maps of negative spin.
- `Spin_maps` returns the conjugate of a missing negative spin for NumPy integer keys too.
- Mapmaking with sky and systematics maps that do not share the same spins.
- `pyyaml` and `toml`, imported by the package, are declared as required dependencies.
- Several fixes of the beam convolution and of `gaussian_circular_beam_blms`.

## History before the changelog

| Date | Milestone |
|------|-----------|
| 2026-10 | $h$-map based exact beam convolution systematics (#2) |
| 2026-06 | `utils` sub-package; sliced $h$-maps and explicit $h$-map readers; pspy support in MASTER |
| 2026-04 | BICEP templates for pointing (HEALPix and CAR) and beam width; MASTER tools for CAR maps; MIT licence |
| 2025-03 | CMB generation tools; `systematics` sub-package |
| 2025-02 | First commit |
