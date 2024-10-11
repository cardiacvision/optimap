"""optimap - An open-source Python toolbox for processing optical mapping and fluorescence imaging data."""

# Check for OpenCV before importing anything else
from importlib.util import find_spec as _find_spec # noqa: I001
if _find_spec("cv2") is None:
    opencv_not_found_error = "\n\n" \
        "ERROR: Unable to import OpenCV, which we require. " \
        "Please install it, e.g. with `pip install opencv-python`. " \
        "See https://cardiacvision.github.io/optimap/main/chapters/getting_started/ for details."  \
        "\n\n"
    raise ImportError(opencv_not_found_error)


from . import activation, image, motion, phase, trace, utils, video
from ._version import __version__, __version_tuple__
from .activation import compute_activation_map
from .image import (
    background_mask,
    foreground_mask,
    interactive_mask,
    invert_mask,
    load_image,
    load_mask,
    save_image,
    save_mask,
    export_image,
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
from .utils import is_verbose, print_bar, print_properties, set_verbose, download_example_data
from .video import (
    export_video,
    export_videos,
    export_video_with_overlay,
    load_metadata,
    load_video,
    save_image_sequence,
    save_video,
    show_video,
    show_video_pair,
    show_videos,
    show_video_overlay,
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

    "extract_traces",
    "compare_traces",
    "select_positions",
    "select_traces",

    "motion_compensate",

    "interactive_mask",
    "background_mask",
    "foreground_mask",
    "invert_mask",

    "compute_phase",
    "compute_activation_map",

    "download_example_data",
    "set_verbose",
    "is_verbose",
    "print_bar",
    "print_properties",
    "detect_apd"
]
