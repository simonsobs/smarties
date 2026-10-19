# SMARTIES package

**SMARTIES**: Spin MAp Reduction Toolkit for systematics Impact EStimation. 

**SMARTIES** is the extension of the map-based approach described in [McCallum et al. (2021)](https://academic.oup.com/mnras/article/513/3/3610/6541854) to simulate systematics at the map-level. 

Tutorial cases are coming in the near future.


## Installation

The easiest way to install and use `smarties` is the following

- clone the repo
    ```
    git clone https://github.com/simonsobs/smarties.git && cd smarties
    ```

- install `smarties`
    ```bash
    pip install .
    ```
    or, for development,
    ```bash
    pip install -e .
    ```

- or, with [uv](https://docs.astral.sh/uv/), create a virtual environment from the pinned
  versions of `uv.lock`
    ```bash
    uv sync --extra test
    uv run python -m pytest
    ```

## Documentation

The documentation (user guide, examples and API reference) lives in [`docs/`](docs/) and is
built with [Zensical](https://zensical.org):

```bash
pip install --group docs   # pip >= 25.1
zensical serve             # live preview at http://localhost:8000
```

## Contact
For any solicitation, please contact `magdy.morshed at fe.infn.it`.

## License
This code is released under the MIT license, which can be found in the [LICENSE](LICENSE) file.


## Citation
If you use `smarties`, please contact `magdy.morshed at fe.infn.it` and consider citing:
```
@unpublished{Morshed:2026,
    author = "The Simons Observatory Collaboration",
    title = "{The Simons Observatory: A framework for efficient map-based simulations of instrumental systematics for CMB surveys}",
    note = {In prep.}
}
@article{Demay:2026efp,
    author = "Demay, Camille and Morshed, Magdy and Gerbino, Martina and Pagano, Luca",
    title = "{Exact map-space beam convolution via spin-mapmaking}",
    eprint = "2610.20231",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "10",
    year = "2026"
}

```
