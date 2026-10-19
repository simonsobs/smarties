# Installation

## Requirements

SMARTIES requires **Python ≥ 3.13**. A dedicated environment is recommended, for instance with
conda:

```bash
conda create -n smarties_env python=3.13
conda activate smarties_env
```

## Installing ducc0 from Source

SMARTIES computes its spherical-harmonic transforms with
[ducc0](https://gitlab.mpcdf.mpg.de/mtr/ducc). The binary wheels on PyPI are compiled for a
generic processor; **compiling ducc0 on your own machine** enables the vector instructions of
your CPU and makes the transforms significantly faster. Install it from source *before*
SMARTIES, so that pip does not pull the binary wheel:

```bash
pip install --no-binary ducc0 ducc0
```

This needs a C++17 compiler and takes a few minutes. Check that the compiled module imports:

```bash
python -c "import ducc0; print(ducc0.__version__)"
```


## Installing SMARTIES

SMARTIES is installed from its repository:

```bash
git clone https://github.com/simonsobs/smarties.git
cd smarties
pip install .
```

For development, install it in editable mode with the test dependencies, and run the tests:

```bash
pip install -e ".[test]"
python -m pytest
```

### With uv

The repository ships a `uv.lock` file that pins every dependency, to install the pacakge with [uv](https://docs.astral.sh/uv/).
From the root of the repository, `uv sync` creates a virtual environment in `.venv` with
SMARTIES installed in editable mode, and `uv run` runs commands in it:

```bash
git clone https://github.com/simonsobs/smarties.git
cd smarties
uv sync --extra test                        # SMARTIES, its dependencies and pytest
uv run python -m pytest
```

Add what you need to the `uv sync` command:

| Option | Installs |
|--------|----------|
| `--extra optional` | `camb` and `matplotlib` |
| `--extra test` | `pytest` |
| `--group docs` | the documentation toolchain (Zensical, mkdocstrings) |
| `--group tutorials` | Jupyter, `camb` and `matplotlib` for the tutorials |
| `--no-binary-package ducc0` | ducc0 compiled from source, [see above](#installing-ducc0-from-source) |

For instance, `uv sync --extra test --no-binary-package ducc0` compiles ducc0 for your CPU;
pass the same options to `uv run`, otherwise it re-synchronises the environment with the
default options. `uv sync --locked` fails instead of updating `uv.lock` if `pyproject.toml`
changed; after editing the dependencies, update the lock file with `uv lock`.


## Dependencies

The required dependencies are installed with SMARTIES:

| Package | Used for |
|---------|----------|
| `numpy`, `scipy`, `opt_einsum` | array algebra, interpolation, tensor contractions |
| `healpy`, `ducc0` | HEALPix pixelisation and spin spherical-harmonic transforms |
| `pixell` | CAR maps (`enmap.ndmap`) and their harmonic transforms |
| `numba` | accelerated filtering of harmonic coefficients |
| `h5py`, `tqdm` | reading $h$-maps, progress bars |
| `pyyaml`, `toml` | YAML / TOML configuration and detector files |

Optional packages enable specific features:

| Package | Feature |
|---------|---------|
| `camb` | CMB power spectra and maps ([`smarties.sky.cmb`][smarties.sky.cmb]) |
| `pymaster` | mode-decoupled power spectra (`uncouple_spectra_*` in [`smarties.utils.master`][smarties.utils.master]) |
| `pspy` | mode-decoupled power spectra ([`uncouple_spectra_pspy`][smarties.utils.master.uncouple_spectra_pspy]) |
| `sotodlib` | Simons Observatory detector metadata ([`smarties.instrument.detectors`][smarties.instrument.detectors]) |
| `matplotlib`, `jupyter` | running the [tutorials](../tutorials/litebird_healpix_tutorial.md) |

## Logging

SMARTIES reports its progress through the standard `logging` module, under the `smarties`
logger. The level is read from the `LOGLEVEL` environment variable (default `INFO`):

```bash
LOGLEVEL=WARNING python my_simulation.py
```

or set from Python, after importing the package:

```python
import logging

logging.getLogger("smarties").setLevel(logging.WARNING)
```

## Building the Documentation

The documentation is built with [Zensical](https://zensical.org) from the `docs` dependency
group (pip ≥ 25.1):

```bash
pip install --group docs
zensical serve   # live preview at http://localhost:8000
```

or, with uv, `uv run --group docs zensical serve`.

<!-- The tutorial pages are rendered from the notebooks of `tutorials/` with
`python tutorials/render_tutorials.py` (add `--execute` to re-run the notebooks first), with the
`tutorials` dependency group: `pip install --group tutorials`, or
`uv run --group tutorials python tutorials/render_tutorials.py --execute`. -->
