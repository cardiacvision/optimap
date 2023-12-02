"""
optimap - An open-source Python toolbox for processing optical mapping and fluorescence imaging data.
"""
try:
    import cv2 as _cv2
except ImportError:
    raise ImportError("\n\nERROR: Unable to import OpenCV, which we require. Please install it, e.g. with `pip install opencv-python`. See https://optimap.readthedocs.io/en/latest/chapters/getting_started/ for details.\n\n")

from . import activation, image, motion, phase, trace, utils, video
from ._version import __version__, __version_tuple__
from .activation import compute_activation_map
from .image import (
    background_mask,
    foreground_mask,
    interactive_mask,
    load_image,
    load_mask,
    save_image,
    show_image,
    show_mask,
)
from .motion import motion_compensate
from .phase import compute_phase
from .trace import (
    compare_traces,
    extract_traces,
    select_positions,
    select_traces,
    show_positions,
    show_traces,
)
from .utils import is_verbose, print_bar, print_properties, set_verbose
from .video import (
    export_video,
    load_metadata,
    load_video,
    save_image_sequence,
    save_video,
)
from .video import (
    play as play_video,
)

__all__ = [
    "__version__",
    "__version_tuple__",

    # submodules
    "motion",
    "phase",
    "plot",
    "utils",
    "video",
    "image",
    "trace",
    "activation",

    # functions
    "load_video",
    "load_metadata",
    "load_image",
    "load_mask",

    "show_image",
    "show_mask",
    "show_positions",
    "show_traces",
    "play_video",

    "save_image",
    "save_image_sequence",
    "save_video",
    "export_video",

    "extract_traces",
    "compare_traces",
    "select_positions",
    "select_traces",

    "motion_compensate",

    "interactive_mask",
    "background_mask",
    "foreground_mask",

    "compute_phase",
    "compute_activation_map",

    "set_verbose",
    "is_verbose",
    "print_bar",
    "print_properties",
]
