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

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )

logger.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())
