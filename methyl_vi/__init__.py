"""methyl_vi setup file. Based off the corresponding file in the scvi-tools repo"""

# Set default logging handler to avoid logging with logging.lastResort logger.
import logging
import warnings
from importlib.metadata import version

from ._constants import METHYLVI_REGISTRY_KEYS

package_name = "methyl_vi"
__version__ = version(package_name)

# Jax sets the root logger, this prevents double output.
scvi_logger = logging.getLogger("scvi")
scvi_logger.propagate = False

# ignore Jax GPU warnings
warnings.filterwarnings("ignore", message="No GPU/TPU found, falling back to CPU.")

__all__ = [
    "METHYLVI_REGISTRY_KEYS",
]
