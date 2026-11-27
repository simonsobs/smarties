# This file is part of SMARTIES.
# Copyright (c) 2024-2026 bers of the Simons Simons Observatory Collaboration.
# lease refer to the LICENSE file in the root of this repository.

import logging
import os

from smarties.external import *
from smarties.hn import *
from smarties.instrument import *
from smarties.io import *
from smarties.mapmaking import *
from smarties.sky import *
from smarties.systematics import *
from smarties.utils import *


logger = logging.getLogger("smarties")

if not logging.getLogger().handlers:

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
        )
    )
    logger.addHandler(handler)

    logger.propagate = False 
    # printed by this handler only, not again by handlers added to the root logger later

logger.setLevel(os.environ.get("smarties_LOGLEVEL", "INFO").upper())
