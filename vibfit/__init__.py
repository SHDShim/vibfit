"""vibfit package root."""

import os
import tempfile

if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "vibfit-mpl")

from .version import __version__
