"""scvi-tools Model classes for methyl_vi."""
from .methyl_anvi import MethylANVI as MethylANVI
from .methyl_vi import MethylVIModel as MethylVI

__all__ = ["MethylVI", "MethylANVI"]
