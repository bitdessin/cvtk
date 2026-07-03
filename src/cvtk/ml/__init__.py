from ._base import *
from . import data
from . import torchutils


# load mmdet utils if the dependencies are available
try:
    import mmdet
    import mmengine
    import mim
    from . import mmdetutils
except ImportError:
    import warnings
    warnings.warn(
        "The environment does not have mim, mmengine, mmcv, and mmdet modules installed. "
        "Functions in the mmdetutils module will not be available. "
        "To ensure full functionality, please install the required dependencies "
        "following the instructions at https://mmdetection.readthedocs.io/en/latest/get_started.html. ",
        ImportWarning
    )
