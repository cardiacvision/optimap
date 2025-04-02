"""optimap - An open-source Python toolbox for processing optical mapping and fluorescence imaging data."""

from . import activation, image, motion, phase, trace, utils, video
from ._version import __version__, __version_tuple__
from .activation import (
    compute_activation_map,
    compute_cv,
    compute_cv_map,
    compute_velocity_field,
    find_activations,
    show_activation_map,
)
from .image import (
    background_mask,
    export_image,
    foreground_mask,
    interactive_mask,
    invert_mask,
    load_image,
    load_mask,
    save_image,
    save_mask,
    show_image,
    show_mask,
)
from .motion import motion_compensate, reverse_motion_compensate
from .phase import compute_phase
from .trace import (
    compare_traces,
    extract_traces,
    select_positions,
    select_traces,
    show_positions,
    show_positions_and_traces,
    show_traces,
)
from .utils import (
    download_example_data,
    is_verbose,
    print_bar,
    print_properties,
    set_verbose,
)
from .video import (
    export_video,
    export_video_with_overlay,
    export_videos,
    load_metadata,
    load_video,
    save_image_sequence,
    save_video,
    show_video,
    show_video_overlay,
    show_video_pair,
    show_videos,
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
    "show_positions_and_traces",
    "play_video",
    "show_video",
    "show_video_pair",
    "show_videos",
    "show_video_overlay",

    "save_image",
    "export_image",
    "save_image_sequence",
    "save_mask",
    "save_video",
    "export_video",
    "export_videos",
    "export_video_with_overlay",

    # trace module
    "extract_traces",
    "compare_traces",
    "select_positions",
    "select_traces",

    # motion module
    "motion_compensate",
    "reverse_motion_compensate",

    "interactive_mask",
    "background_mask",
    "foreground_mask",
    "invert_mask",

    # activation module
    "find_activations",
    "compute_activation_map",
    "show_activation_map",
    "compute_cv",
    "compute_cv_map",
    "compute_velocity_field",

    # phase module
    "compute_phase",

    # utils module
    "download_example_data",
    "set_verbose",
    "is_verbose",
    "print_bar",
    "print_properties",
]
