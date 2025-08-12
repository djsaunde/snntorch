from ._version import __version__
from ._neurons import *
from ._layers import *
# from .export_nir import export_to_nir
# from .import_nir import import_from_nir

# FSDP2 support for distributed training (optional import)
try:
    from . import distributed
    from . import trainer
    __all_distributed__ = ['distributed', 'trainer']
except ImportError:
    # FSDP2 dependencies not available
    __all_distributed__ = []