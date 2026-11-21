# Logs

SMARTIES reports what it does through the standard [`logging`](https://docs.python.org/3/library/logging.html) module, with two levels of messages:

- `INFO`: progress — files read, chunks processed, time spent;
- `WARNING`: situations worth checking — no systematics maps given, $h$-map values replaced by the nearest detector in the interpolation, missing files of spherical derivatives.

Errors are raised as exceptions, not logged. The messages are printed on the standard error, with the time, the level and the module that emitted them:

```text
19-Nov-26 16:29:25 WARNING smarties.mapmaking: No systematics maps provided, assuming no systematics
19-Nov-26 16:29:25 INFO smarties.mapmaking: Mapmaking per chunk: 48896 pixels, 12416 h-map pixels, 8 detectors, 8 chunks
19-Nov-26 16:29:25 INFO smarties.mapmaking: Processing chunk with 6064 pixels and 1552 h-map pixels...
```

Each module logs under its own name (`smarties.mapmaking`, `smarties.io`, `smarties.utils.interpolation`, ...), below the `smarties` logger.

## Setting the Level

The level is read from the `smarties_LOGLEVEL` environment variable when SMARTIES is imported (default `INFO`):

```bash
smarties_LOGLEVEL=WARNING python my_simulation.py  # warnings only
```

From Python, the level can be changed at any time after the import:

```python
import logging

import smarties

logging.getLogger("smarties").setLevel(logging.WARNING)  # warnings only
logging.getLogger("smarties").setLevel(logging.INFO)  # back to the default
```

The level of the `smarties` logger applies to all the modules of the package.

## Silencing the Logs

| To silence | Use |
|------------|-----|
| all the messages of SMARTIES | `smarties_LOGLEVEL=ERROR`, or `logging.getLogger("smarties").setLevel(logging.ERROR)` |
| the progress messages only | `smarties_LOGLEVEL=WARNING`, or `logging.getLogger("smarties").setLevel(logging.WARNING)` |
| one module, e.g. the warnings of the interpolation, repeated for each chunk | `logging.getLogger("smarties.utils.interpolation").setLevel(logging.ERROR)` |
| all the libraries, SMARTIES included | `logging.disable(logging.WARNING)`, undone with `logging.disable(logging.NOTSET)` |

!!! note
    Use the level to silence the package: `logging.getLogger("smarties").disabled = True`
    only disables the `smarties` logger itself, not the loggers of its modules.

## Writing the Logs to a File

Replace the handler printing the messages by a file handler:

```python
logger = logging.getLogger("smarties")
logger.handlers.clear()  # without this line, the messages are printed and written to the file
handler = logging.FileHandler("smarties.log")
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logger.addHandler(handler)
```
## Progress Bars

The readers of litebird_sim $h$-maps ([`read_lbs_h_maps`][smarties.io.read_lbs_h_maps], `build_mask_from_lbs_h_maps`) show progress bars with tqdm, which do not go through `logging`. Disable them with the environment variable of tqdm:

```bash
TQDM_DISABLE=1 python my_simulation.py
```
