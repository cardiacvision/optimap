"""
optimap - An open-source Python toolbox for processing optical mapping and fluorescence imaging data.
"""
from . import activation, motion, phase, trace, utils, video, image
from ._version import __version__, __version_tuple__
from .activation import compute_activation_map
from .video import (
    load_video,
    load_metadata,
    save_image_sequence,
    save_video,
    export_video,
    play as play_video,
)
from .image import (
    show_image,
    show_mask,
    load_image,
    load_mask,
    save_image,
    background_mask,
    foreground_mask
)
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
from .motion import motion_compensate

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
    "set_verbose",
    "is_verbose",
    "print_bar",
    "print_properties",
    "compute_phase",
    "compute_activation_map",
    "show_positions",
    "show_traces",
    "extract_traces",
    "compare_traces",
    "select_positions",
    "select_traces",
    "show_image",
    "show_mask",
    "load_image",
    "load_mask",
    "background_mask",
    "foreground_mask",
    "load_video",
    "load_metadata",
    "save_image",
    "save_image_sequence",
    "save_video",
    "export_video",
    "play_video",
    "motion_compensate",
]
